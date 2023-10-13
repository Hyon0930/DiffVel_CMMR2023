
import csv
import datetime
import logging
import os
import pickle
import sys
import time

import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
import numpy as np
import hydra

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics



from model.diffwave import ClassifierFreeDiffRoll

import src.evaluation as evaluation
from src.utilities import TargetProcessor, int16_to_float32
import model as Model

from task.diffusion import SpecRollDiffusion, q_sample, linear_beta_schedule
from model.diffwave import ClassifierFreeDiffRoll

def unit_data_dict_gen(start_time, sample_rate, segment_samples, hf, target_processor, shift):

    data_dict = {}

    start_time += shift
    start_sample = int(start_time * sample_rate)
    end_sample = start_sample + segment_samples

    if end_sample >= hf['waveform'].shape[0]:
        # start_sample -= segment_samples
        # end_sample -= segment_samples
        pad_len =  segment_samples - len(hf['waveform'][start_sample : end_sample])
        print("remainder", len(hf['waveform'][start_sample : end_sample]))
        print("pad_len", pad_len)
        waveform_padded = np.pad(hf['waveform'], (0, pad_len), mode='constant')
        waveform = waveform_padded[start_sample : end_sample]
        if waveform.shape[0] != segment_samples:
            print("waveform.shape", waveform.shape)
            exit()

        waveform = int16_to_float32(waveform)
    else:
        waveform = int16_to_float32(hf['waveform'][start_sample : end_sample])

    
    data_dict['waveform'] = waveform
    midi_events = [e.decode() for e in hf['midi_event'][:]]
    midi_events_time = hf['midi_event_time'][:]

    

    # Process MIDI events to target
    (target_dict, note_events, pedal_events) = \
        target_processor.process(start_time, midi_events_time, 
            midi_events, extend_pedal=True, note_shift=0)
    
            
    for key in target_dict.keys():
        data_dict[key] = target_dict[key]
    
    
    return data_dict



def enframe(x, segment_samples):
    """Enframe long sequence to short segments.

    Args:
        x: (1, audio_samples)
        segment_samples: int

    Returns:
        batch: (N, segment_samples)
    """
    assert x.shape[1] % segment_samples == 0
    batch = []

    pointer = 0
    while pointer + segment_samples <= x.shape[1]:
        batch.append(x[:, pointer : pointer + segment_samples])
        pointer += segment_samples // 2

    batch = np.concatenate(batch, axis=0)
    return batch

def audio_to_segment(data_dict, segment_samples):
    audio = data_dict['waveform']
    audio = audio[None, :]  # (1, audio_samples)
    # Pad audio to be evenly divided by segment_samples
    audio_len = audio.shape[1]
    pad_len = int(np.ceil(audio_len / segment_samples)) \
        * segment_samples - audio_len
    audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)
    # Enframe to segmentss
    segments = enframe(audio, segment_samples)
    #(N, segment_samples)

    return segments


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """

    np_data_dict = {}
    for key in list_data_dict[0].keys():
        #print("key",key)
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict], dtype=np.float32)
    
    return np_data_dict

def create_pkl(pkl_path, content):

    obj_pkl = open(pkl_path, "wb")
    pickle.dump(content, obj_pkl)
    obj_pkl.close()


def get_pkl_output_target(pkl_file):

    pkl_file = open(pkl_file,'rb')
    dump_list = pickle.load(pkl_file)
            
    output_dict_list = dump_list[0]
    target_dict_list = dump_list[1]

    return output_dict_list, target_dict_list


def adjust621_640(array_dict):
    #adjust for test data
    necessarry_keys = ["frame_roll", "velocity_roll", "onset_roll", "pedal_frame_roll"]
    for key in array_dict:
        if key in necessarry_keys:
            if key == "pedal_frame_roll":
                array = array_dict[key]
                last_4_elements = array[-4:, :]
                extended_array = np.concatenate((array, last_4_elements), axis=1)
                array_dict[key] = extended_array
            else:
                array = array_dict[key]
                last_4_elements = array[:, -4:, :]
                extended_array = np.concatenate((array, last_4_elements), axis=1)
                array_dict[key] = extended_array
            

    return array_dict


def error_gen(model, hdf5_file, shift, **kwargs):

    """
    Used as in an iterator; for loop on each file, this function to get all the test data to be fed into model for reference/forward function and get estimation of MIDI vel. 
 
        Args:
            hdf5_file
            segment_seconds
            test_hopseconds

        extracts
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        
        Returns: 
            Collated data_dict for one music. 

        Assert Test: 
            assret duration == len(wave)/sampling freq 

    """

    start_time = 0
    data_dict = {}

    output_dict_list = []
    target_dict_list = []

    separated_path = hdf5_file.split("/")
    data_dict['audio_name']=separated_path[-1]

    
    segment_samples = kwargs["segment_samples"]
    target_processor = TargetProcessor(kwargs["segment_seconds"], kwargs["frames_per_second"], 21, 88)
    
    # segment_seconds = wave_length / config.sample_rate

    with h5py.File(hdf5_file, 'r') as hf:

        
        while start_time < hf.attrs["duration"]:

            
            data_dict = unit_data_dict_gen(start_time, kwargs["sampling_rate"], segment_samples, hf, target_processor, 0) # groud truth; no shifted data in time 
            shifted_data_dict = unit_data_dict_gen(start_time, kwargs["sampling_rate"], segment_samples, hf, target_processor, shift)

            audio_segment= audio_to_segment(data_dict, segment_samples)
            print("audio_segment", audio_segment.shape)
            data_dict = [data_dict]
            shifted_data_dict = [shifted_data_dict]
            
            np_data_dict = collate_fn(data_dict)
            shifted_np_data_dict = collate_fn(shifted_data_dict)

            np_data_dict = adjust621_640(np_data_dict)
            shifted_np_data_dict = adjust621_640(shifted_np_data_dict)
            
            tensor_onset = torch.from_numpy(np_data_dict['onset_roll'])
            tensor_frame = torch.from_numpy(np_data_dict['frame_roll'])

            shift_tensor_onset = torch.from_numpy(shifted_np_data_dict['onset_roll'])
            shift_tensor_frame = torch.from_numpy(shifted_np_data_dict['frame_roll'])
            
            print("tensor_frame", tensor_frame.shape)
            print("shift_tensor_frame", shift_tensor_frame.shape)


            batch = {
                "audio": audio_segment,
                "frame": shift_tensor_frame
            }

            output = evaluation.forward(model, batch, 1)
            output["velocity_output"] = np.asarray(output["velocity_output"])
            print("output", type(output["velocity_output"]))
            print("np_data_dict", type(np_data_dict["velocity_roll"]))

            output_dict_list.append(output)
            target_dict_list.append(np_data_dict)
            start_time += kwargs["test_hopseconds"]

        return output_dict_list, target_dict_list



def eval_from_list(output_dict_list, target_dict_list):
    
    result_mean = np.empty((0,1),float)
    result_std = np.empty((0,1),float)  
    score_error = np.empty((0,88), float)

    num_note = 0
    for i, target_dict_segmentseconds in enumerate(target_dict_list):
        output_dict_segmentseconds = output_dict_list[i]
        segment_error, num_onset = evaluation.note_level_l1_per_window(output_dict_segmentseconds, target_dict_segmentseconds)
        score_error = np.append(score_error, segment_error, axis=0)
        num_note += num_onset
    mean_error = np.sum(score_error)/num_note
    std_error = score_error[score_error!=0].std()
    return mean_error, std_error


def eval_from_pkl_wrap(pkl_dir):

    pkls = os.listdir(pkl_dir)
    for pkl in pkls:

        pkl_fullpath = os.path.join(pkl_dir, pkl)
        output_dict_list, target_dict_list = get_pkl_output_target(pkl_fullpath)
        result_mean = []
        for i, target_dict_segmentseconds in enumerate(target_dict_list):
            output_dict_segmentseconds = output_dict_list[i]
            mean_note_error_l1 = evaluation.note_level_l1_per_window(output_dict_segmentseconds, target_dict_segmentseconds)
            result_mean.append(mean_note_error_l1)
        
        mean_notelevel_l1_error = sum(result_mean) / len(result_mean)

    return 0 #mean_l1_error, std_l1_error

def file_schema(checkpoint_path):
    model_schema = checkpoint_path.split("/")
    model_info = model_schema[-7]
    return model_info

def error_histogram(error_profile):
    errors = []
    for note_error_profile in error_profile:
        errors.append(note_error_profile["note_error"])

    return errors 

def chart_simulnotes(error_profile):
    """_summary_
        - number of simultaneous note vs average error with min,max chart
            - x: number of simultaniopus nots, y: violin chart, note ratio(percentage of the x axis in the score)
    Args:
        error_profile (_type_): _description_

        note_error_profile = {"pitch":0-88, "duration":[start, end], 
                                "note_error": double, "ground truth":0-127, 
                                "estimation":double, "pedal_check": True/False, 
                                "simultaneous_notes":int, "classification_check":True/False} 
                               
    """

    simultaneous_note_dict = {}

    for error in error_profile:
        key = str(error["simultaneous_notes"])
        if key in simultaneous_note_dict:
            simultaneous_note_dict[key].append(error["note_error"])
        else:
            simultaneous_note_dict[key] = [error["note_error"]]

    sorted_dict = {key: value for key, value in sorted(simultaneous_note_dict.items())}
    boxplot_data = []
    for key, value in sorted_dict:
        print("key", key)

    return 0



def noise_check(output_dict_list):
    
    means = []
    stds = []
    for output_dict in output_dict_list:
        output_np = np.array(output_dict['velocity_output'])
        mean = np.mean(output_np[:, 80:88])
        std = np.std(output_np[:, 80:88])
        means.append(mean)
        stds.append(std)
        print("Mean:", mean)
        print("Standard deviation:", std)

    return means, stds

    

@hydra.main(config_path="config", config_name="inference")
def main(cfg):

    #model = load_model()
    h5_files = os.listdir(cfg.SMD_data_root)
    model_info = "diffvel_baseline"  #diffvel_film, diffvel_sinc_baseline, diffvel_sincfilm 


    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder_name = cfg.inference_result_path + model_info
    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)

    error_profile_list = []
    all_score_error = []

    segment_seconds = cfg.sequence_length/cfg.sampling_rate
    segment_samples = int(cfg.sampling_rate * segment_seconds)
    sampling_rate = cfg.sampling_rate
    frames_per_second = cfg.frames_per_second
    
    cfg_dict = {
        "segment_seconds" : segment_seconds, 
        "segment_samples" : segment_samples,
        "test_hopseconds" : segment_seconds,
        "sampling_rate" : sampling_rate, 
        "frames_per_second" : frames_per_second
    }


    model = getattr(Model, cfg.model.name).load_from_checkpoint(cfg.comp_modelckpt, frame_threshold=0.5, sampling=cfg.task.sampling)


    for shift in cfg.unaligned_shift:
        with open(result_folder_name + "/diffvel_base_shift_" + str(shift) +"-"+ current_time + ".csv", "w") as result:
            writer=csv.writer(result, delimiter=',' , lineterminator='\n')
            fields = ["g_means", "g_stds", "test_h5", "frame_max_error", "std_max_error", "mean_error_correct_detection", "std_error_correct_detection", "average_precision_score", "f1_score", "precision_score", "recall_score", "precision_support", "recall_support", "f1_support"]
            writer.writerow(fields)
            
            for test_h5 in h5_files:
                print("test_h5", test_h5)
                pkl_name, _ = test_h5.split(".")
                test_h5_path = os.path.join(cfg.SMD_data_root, test_h5)
                result_filename = pkl_name + "_" + str(shift) + "_" + current_time
                pair_result_filename = pkl_name + "_pair_" + str(shift) + "_" + current_time

                npy_error_dict_fullpath = os.path.join(cfg.inference_result_path, result_filename)
                pair_dict_fullpath = os.path.join(cfg.pair_pkl_dir, pair_result_filename)

                output_dict_list, target_dict_list = error_gen(model, test_h5_path, shift, **cfg_dict)

                means, stds = noise_check(output_dict_list)
                
                pairs = [output_dict_list, target_dict_list]
                np.asarray(pairs)
                np.save(pair_dict_fullpath, pairs, allow_pickle=True)
                
                # classification_error = ["average_precision_score, f1_score, precision_score, recall_score, precision_support, recall_support, f1_support"]
                frame_max_error, std_max_error, error_profile, precision_ave, f1, precision, recall, frame_precision, frame_recall, frame_f1 = evaluation.gt_to_note_list(output_dict_list, target_dict_list)
                mean_error_correct_detection, std_error_correct_detection = evaluation.error_on_correct_detection(error_profile)
                #onset_error, std_error = eval_from_list(output_dict_list, target_dict_list)
                row = (means, stds, test_h5, frame_max_error, std_max_error, mean_error_correct_detection, std_error_correct_detection, precision_ave, f1, precision, recall, frame_precision, frame_recall, frame_f1)
                writer.writerow(row)
                
                np.save(npy_error_dict_fullpath, error_profile, allow_pickle=True)
                
                all_score_error += error_histogram(error_profile)
                
        plt.hist(all_score_error, bins=[i for i in range(0, int(max(all_score_error))+1)])
        plt.xlabel('Error Value')
        plt.ylabel('Number of Notes')
        plt.title('Distrobution of Error')
        plt.savefig('error_dist.png')
        plt.close()


if __name__ == "__main__":
    main()




