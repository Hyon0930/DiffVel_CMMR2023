import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score , average_precision_score, precision_recall_fscore_support
import time
import copy
import datetime
 

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

def append_to_dict(midi_dict, key, value):
    
    if key in midi_dict.keys():
        midi_dict[key].append(value)
    else:
        midi_dict[key] = [value]

def note_level_l1_per_window(output_per_segmentseconds, target_per_segmentseconds):
    
    """_summary_
    Args:
        output_per_window (list): list having output dictionary [{output_vel:values}]
        target_per_window (list): _description_
        mask (_type_): _description_
    Returns:
        _type_: _description_
    """
    error_segemnt_per_segment = np.empty((0,1), float)
    spot_check = 0
    num_notes = 0
    accum_error = 0
    segment_error = np.empty((0,88), float)

    for nth_frame, output_dict in enumerate(output_per_segmentseconds):
        
        output_vel_frame = output_per_segmentseconds['velocity_output'][nth_frame]
        gt_vel_frame = target_per_segmentseconds['velocity_roll'][nth_frame] #gt_vel_frame.shape = (201, 88)
        gt_frame_frame = target_per_segmentseconds['frame_roll'][nth_frame]
        gt_onset_frame = target_per_segmentseconds['onset_roll'][nth_frame]
        
        """
        output_vel_frame.shape, gt_vel_frame.shape, gt_frame_frame.shape  -> (201,88) 100 frames/sec, 88keys
        """
        

        if np.count_nonzero(gt_onset_frame) != 0:
            #solution1 gets midi velocity where onset happends only. 
            sol1_frame = np.multiply(output_vel_frame, gt_onset_frame)
            #ignore offset values.
            gt_frame = np.multiply(gt_vel_frame, gt_onset_frame)
            #back to midi scale 0-128
            sol1_frame = sol1_frame * 128
            sol1_note_error = np.abs(np.subtract(sol1_frame, gt_frame))
            num_notes += np.count_nonzero(gt_onset_frame)


            segment_error = np.append(segment_error, sol1_note_error, axis=0)

            
            # if spot_check%10 == 0:
            #     gen_inf_image(gt_vel_frame, output_vel_frame)
            
            spot_check += 1

    return segment_error, num_notes # mean_error_segemnt_per_segment, std_error_segemnt_per_segment

#Note level analysis------------------------------------------------------------------------

def gt_to_note_list(output_dict_list, score_note_frame):
    """_summary_
    frame structure; (time x keys), e.g. 201x88
    Args:
        score_note_frame (_type_): _description_
    """
    score = np.empty((0,88), float)
    pedal = np.empty((0,88), float)
    estimation = np.empty((0,88), float)

    for i, target_dict_segmentseconds in enumerate(score_note_frame):
        output_dict_segmentseconds = output_dict_list[i]
        
        for nth_frame, gt_seg in enumerate(output_dict_segmentseconds):
            #gt_velframe = np.squeeze(gt_seg['velocity_roll'], axis=0)
            gt_velframe = target_dict_segmentseconds['velocity_roll'][nth_frame]  #gt_vel_frame.shape = (201, 88)
            gt_pedal = target_dict_segmentseconds['pedal_frame_roll'][nth_frame]
            # print("gt_velframe", gt_velframe.shape)
            
            output_vel_frame = output_dict_segmentseconds['velocity_output'][nth_frame]
            # print("output_vel_frame", output_vel_frame.shape)
            score = np.append(score, gt_velframe, axis=0)
            pedal = np.append(pedal, gt_pedal)
            estimation = np.append(estimation, output_vel_frame, axis=0)

            # if (nth_frame%10 == 0):
            #     gen_inf_image(gt_velframe, output_vel_frame)
            #     print("gt-inf image generated")
    
    # print("score", score.shape)
    # print("estimation", estimation.shape)
    # print("pedal", pedal.shape)

    precision_ave, f1, precision, recall, frame_precision, frame_recall, frame_f1 = classification_error(copy.deepcopy(score), copy.deepcopy(estimation))

    score = np.transpose(score)
    estimation = np.transpose(estimation)

    score_sound_profile = get_midi_sound_profile(score)

    error_profile = []
    accum_error = []

    for note_profile in score_sound_profile: 
        vel_est = estimation[note_profile["pitch"]][note_profile["duration"][0]:note_profile["duration"][1]]
        vel_est[vel_est<= 0.0001] = 0
        if sum(vel_est) > 0:
            classification_check = True
        else: 
            classification_check = False


        max_estimation = max(vel_est) * 128
        notelevel_error = abs(max_estimation - note_profile["velocity"])


        sim_note_count = num_simultaneous_notes(note_profile, score)
        pedal_onoff = pedal_check(note_profile, pedal)

        note_error_profile = {"pitch":note_profile["pitch"], "duration":note_profile["duration"], 
                                "note_error":notelevel_error, "ground truth":note_profile["velocity"], 
                                "estimation":max_estimation, "pedal_check": pedal_onoff, 
                                "simultaneous_notes":sim_note_count, "classification_check":classification_check} 
                               
        error_profile.append(note_error_profile)
        accum_error.append(notelevel_error)

    frame_max_error = np.mean(accum_error)
    std_max_error = np.std(accum_error)

    return frame_max_error, std_max_error, error_profile, precision_ave, f1, precision, recall, frame_precision, frame_recall, frame_f1


def error_on_correct_detection(error_profile):

    error_accum = []

    for note_error_dict in error_profile:
        if note_error_dict["classification_check"]:
            error_accum.append(note_error_dict["note_error"])

    error_correct_detection = np.mean(error_accum)
    std_correct_detection = np.std(error_accum)


    return error_correct_detection, std_correct_detection


def pedal_check(note_profile, pedal):

    check = pedal[note_profile["duration"][0]:note_profile["duration"][1]]
    if sum(check) > 0: 
        return True
    else:
        return False

def classification_error(score, estimation):

    score[score > 0] = 1
    estimation[estimation > 0.0001] = 1
    estimation[estimation <= 0.0001] = 0\

    precision_ave = average_precision_score(score.flatten(), estimation.flatten(), average='macro')

    f1 = f1_score(score.flatten(), estimation.flatten(), average="macro")
    precision = precision_score(score.flatten(), estimation.flatten(), average="macro")
    recall = recall_score(score.flatten(), estimation.flatten(), average="macro")

    tmp = precision_recall_fscore_support(score.flatten(), estimation.flatten())
    frame_precision = tmp[0][1]
    frame_recall = tmp[1][1]
    frame_f1 = tmp[2][1]

    # print("average_precision_score, f1_score, precision_score, recall_score, precision_support, recall_support, f1_support", precision_ave, f1, precision, recall, frame_precision, frame_recall, frame_f1)
    # classification_acc = [precision_ave, f1, precision, recall, frame_precision, frame_recall, frame_f1]

    return precision_ave, f1, precision, recall, frame_precision, frame_recall, frame_f1

def gen_inf_image(score, estimation):
    """_summary_

    Args:
        score (np.array):  shape (201, 88)
        estimation np.array):  shape (201, 88)

    Returns:
        _type_: _description_
    """
    score = np.transpose(score)
    estimation = np.transpose(estimation)

    fig,axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
    im0 = axs[0].imshow(score, cmap='hot', aspect='auto')
    axs[0].title.set_text('Groud Truth')
    im1 = axs[1].imshow(estimation, cmap='hot', aspect='auto')
    axs[1].title.set_text('Estimation')
    #fig.colorbar(im0, orientation='vertical')
    axs[0].set(xlabel="time",ylabel="88 piano keys")
    axs[1].set(xlabel="time",ylabel="88 piano keys")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    current_directory = os.getcwd()
    dir_path = os.path.join(current_directory, "output/")
    
    file_name = timestr + ".png"
    file_path = os.path.join(dir_path, file_name)

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        
    fig.savefig(file_path)
    plt.close(fig)

    return 0 



def num_simultaneous_notes(note_profile, score):
    sim_note_count = 0
    for key in range(0,88):
        sim_note = score[key][note_profile["duration"][0]:note_profile["duration"][1]]
        if np.sum(sim_note) > 0:
            sim_note_count += 1

    return sim_note_count

def get_midi_sound_profile(midi_vel_roll):
    """_summary_
    
    Args:
        midi_roll (array): (time x 88) array of MIDI roll contrains either note velocity information
    Return:
        sound_profile(list): ([pitch, velocity, interval([x1, x2])], ([pitch, velocity, interval([x3, x4])] , ..., ([pitch, velocity, interval([xn-1, xn])])
                                where MIDI roll sound. 
    """
    sound_profile = []

    for pitch, key in enumerate(midi_vel_roll):
        
        # example
        # [0, ... , 0, 100, ..., 100, 0, ..., 0, 80, ..., 80, 0, ..., 0]
        # iszero = np.concatenate(([0], np.equal(key, 0).view(np.int8), [0]))
        #-> [0, 1, ..., 1, 0, sound, 0, 1, ..., 1, 0, sound, 0, 1, ..., 1, 0]
        # absdiff = np.abs(np.diff(iszero))
        #-> [1, 0, ..., 1, 0, sound, 1, 0, ..., 1, 0, sound, 1, 0, ..., 1]
        # ranges = np.where(absdiff == 1)[0].reshape(-1, 2) 
        # [[0,x1], [x2, x3], ..., [xn, xm]] which are originally silent. 
        # romove edge and reshape: [[x1, x2], [x3, x4], ...[xn-1, xn]] which are durations of sound. 

        iszero = np.concatenate(([0], np.equal(key, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0] # Might be done just removing first element and reshaape 
        index = [0,-1]
        temp = np.delete(ranges, index)
        sound_durations = temp.reshape(-1, 2)
        #print("ranges", pitch, sound_interval)

        if len(sound_durations) != 0 : 
            for duration in sound_durations: 
                vel = midi_vel_roll[pitch, duration[0]]
                key_profile = {"pitch":pitch, "velocity": vel, "duration": duration}
                #print("key_profile", key_profile)
                sound_profile.append(key_profile)
        

    
    return sound_profile

def forward(model, batch, batch_size):
    """Forward data to model in mini-batch.
    
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int

    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
        
    """

    x=batch["audio"]
    output_dict = {}
    param = model.parameters()
    device = next(param).device
    pointer = 0
    model.eval()
    idx = 0
    while True:
        if pointer >= len(x):
            break
        idx += 1
        batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size
        batch["audio"] = batch_waveform
        with torch.no_grad():
            batch_output = model.test_step_custom(batch, idx)
            batch_output = batch_output.squeeze()
            print("batch_output check", batch_output.shape, type(batch_output))
            plt.figure(figsize=(10, 6))  # Adjust the figure size for better visualization
            plt.imshow(batch_output, aspect='auto', cmap='viridis', origin='lower')  # 'cmap' sets the colormap, and 'aspect' adjusts the aspect ratio
            plt.colorbar() 
            plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')       
        
        print("batch_output", batch_output.shape)
        append_to_dict(output_dict, "velocity_output", batch_output.data)


    return output_dict

