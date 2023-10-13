import copy
import csv
import datetime
import os, sys
import time
import numpy as np
import argparse

import matplotlib.pyplot as plt

import matplotlib.ticker as mtick
import src.evaluation as evaluation
import task.error_visualizer as error_vis


def create_output_dir(BASE_DIR):
    # List of directories to create
    paths = {
        'pair_pkl': os.path.join(BASE_DIR, "outputs/pair_pkl"),
        'error_profile': os.path.join(BASE_DIR, "outputs/error_profile"),
        'noise_img_gen': os.path.join(BASE_DIR, "outputs/noise_visualization"),
        'chart_folder': os.path.join(BASE_DIR, "outputs/chart"),
        'result_folder': os.path.join(BASE_DIR, "outputs/result"),
        'training_set': os.path.join(BASE_DIR, "outputs/maestro")
    }

    # Create directories if they do not exist
    for key, dir_path in paths.items():
        os.makedirs(dir_path, exist_ok=True)

    return paths


def z_th_gnoise(pairs, z_score):

    output_dict_list = pairs[0]
    gt_dict_list = pairs[1]

    outlier_th_list = []
    
    for i, outputs in enumerate(output_dict_list):
        gt_dict = gt_dict_list[i]
        
        zero_elements_in_gt = outputs["velocity_output"][gt_dict["velocity_roll"] == 0]


        mean_noise = np.mean(zero_elements_in_gt)
        std_noise = np.std(zero_elements_in_gt)
        z_score_outlier = [i for i in zero_elements_in_gt if (i-mean_noise)/std_noise > z_score]
        max_outlier = np.max(z_score_outlier)
        min_outlier = np.min(z_score_outlier)
        outlier_th_list.append(min_outlier)

        
    trios = [output_dict_list, gt_dict_list, outlier_th_list]
    
    return trios


def rm_noise(trios):

    #trios = [output_dict_list, gt_dict_list, outlier_th_list]
    output_dict_list = trios[0]
    gt_dict_list = trios[1]
    outlier_th_list = trios[2]
    cleaned_output_list = []

    for i, outputs in enumerate(output_dict_list):
        target_noise_th = outlier_th_list[i]
        
        vel_est = outputs["velocity_output"]
        vel_est[vel_est <= target_noise_th] = 0
        outputs["velocity_output"] = vel_est
        cleaned_output_list.append(outputs)
    
    return [cleaned_output_list, gt_dict_list]

def rescale_linear01(pairs): 

    output_dict_list = pairs[0]
    gt_dict_list = pairs[1]
    normalised_output_list = []
    all_output = np.empty((0,88),float)
    copied_output_list = copy.deepcopy(output_dict_list)
    for outputs in copied_output_list:
        outputs["velocity_output"] = np.squeeze(outputs["velocity_output"])

        all_output = np.append(all_output, outputs["velocity_output"], axis=0)
    
    min_value = np.min(all_output)
    max_value = np.max(all_output)

    for outputs in output_dict_list:
        # Normalize the array to the range [0, 1]
        outputs["velocity_output"] = (outputs["velocity_output"]- min_value) / (max_value - min_value)

        normalised_output_list.append(outputs)

    return normalised_output_list, gt_dict_list


def rescale_log01(pairs):

    output_dict_list = pairs[0]
    gt_dict_list = pairs[1]
    normalised_output_list = []
    all_output = np.empty((0,88),float)
    copied_output_list = copy.deepcopy(output_dict_list)
    for outputs in copied_output_list:
        outputs["velocity_output"] = np.squeeze(outputs["velocity_output"])
        all_output = np.append(all_output, outputs["velocity_output"], axis=0)
    
    min_value = np.min(all_output)
    max_value = np.max(all_output)

    for outputs in output_dict_list:
        # Normalize the array to the range [0, 1]
        outputs["velocity_output"] = np.exp((np.log(outputs["velocity_output"]) - np.log(min_value)) / (np.log(max_value) - np.log(min_value)) * (np.log(1) - np.log(0)) + np.log(0))
        
        normalised_output_list.append(outputs)

    return normalised_output_list, gt_dict_list


def get_args():
    parser = argparse.ArgumentParser(description='Create directories for a project.')
    parser.add_argument('--output_dir', type=str, help='The base directory path for output.')
    parser.add_argument('--npy_pair_dir', type=str, help='Directory having npy files of pair of model estimation and ground truth.')
    args = parser.parse_args()

    return args

def main(args):

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    BASE_DIR = args.output_dir
    output_dirs = create_output_dir(BASE_DIR)
    
    inf_pny_list = os.listdir(args.npy_pair_dir)
    z_scores = [3]
    gt_denoise_list = []
    z_score0_denoise_list = []
    z_score1_denoise_list = []
    z_score2_denoise_list = []
    z_score3_denoise_list = []

    

    with open(output_dirs["result_folder"] + "/diffvel_film-" + current_time + ".csv", "w") as result:
        
        writer=csv.writer(result, delimiter=',' , lineterminator='\n')
        new_fields = ["test_h5", "frame_max_error", "std_max_error", "mean_error_correct_detection", "std_error_correct_detection", "recall_score",  "recall_support", "z_score"]
        
        writer.writerow(new_fields)

        for z_score in z_scores:

            for pair_npy in inf_pny_list: 
                print("pair_npy", pair_npy)
                music_name, rem = pair_npy.split("_pair_")
                
                if rem.endswith(".npy"):

                    pair_npy_file = os.path.join(args.npy_pair_dir, pair_npy)
                    pairs = np.load(pair_npy_file, allow_pickle=True)
                    # pairs = [output_dict_list, target_dict_list]
                    trios = z_th_gnoise(pairs, z_score)
                    # trios = [output_dict_list, gt_dict_list, outlier_th_list]

                    assertion = pairs[0] == trios[0]
                    if assertion.all() == False:
                        print("ourder is not equial")

                    if z_score == 0:
                        z_score0_denoise_list.append(pairs[0])
                        gt_denoise_list.append(pairs[1])

                        normalised_pairs = rescale_linear01(pairs)

                    else:
                        cleaned_pairs = rm_noise(trios)
                        if z_score == 1:
                            z_score1_denoise_list.append(cleaned_pairs[0])
                        elif z_score == 2: 
                            z_score2_denoise_list.append(cleaned_pairs[0])
                        elif z_score == 3:
                            z_score3_denoise_list.append(cleaned_pairs[0])

                        normalised_pairs = rescale_linear01(cleaned_pairs)
                
                    frame_max_error, std_max_error, error_profile, precision_ave, f1, precision, recall, frame_precision, frame_recall, frame_f1 \
                        = evaluation.gt_to_note_list(normalised_pairs[0], normalised_pairs[1])

                    
                    mean_error_correct_detection, std_error_correct_detection = evaluation.error_on_correct_detection(error_profile)
                    #onset_error, std_error = eval_from_list(output_dict_list, target_dict_list)
                    #new_fields = ["test_h5", "frame_max_error", "std_max_error","recall_score",  "recall_support", "z_score"]
                    row = (music_name, frame_max_error, std_max_error, mean_error_correct_detection, std_error_correct_detection, recall, frame_recall, z_score)
                    writer.writerow(row)
                    # error_profile_path = os.path.join(error_profile_dir, "error_prof" + pair_npy) 
                    # np.save(error_profile_path, error_profile, allow_pickle=True)


    # for i in range(0, len(gt_denoise_list)):
    #     noise_visualization(gt_denoise_list[i], z_score0_denoise_list[i], z_score1_denoise_list[i], z_score2_denoise_list[i], z_score3_denoise_list[i], noise_img_gen_dir)

    # error_analysis(error_profile_dir, PATH_TO_TRAININGSET_NPY)





if __name__ == "__main__":


    args = get_args()
    main(args)  




# def error_analysis(error_prof_path, training_npy_path, outout_dir):

#     error_profs = os.listdir(error_prof_path)
#     training_npy_list = os.listdir(training_npy_path)
#     all_errors_correct_classification = []
#     all_training_notes =[]

#     for npy_file in error_profs:
#         if npy_file.endswith(".npy"):
#             error_profile = np.load(os.path.join(error_prof_path, npy_file), allow_pickle=True)
#             correct_classification, miss_classification = error_vis.missclassification_sort(error_profile)
#             all_errors_correct_classification.append(correct_classification)

#     for npy_file in training_npy_list:
#         print("npy_file", npy_file)
#         trainig_note_profile = np.load(os.path.join(training_npy_path, npy_file), allow_pickle=True)
#         all_training_notes.append(trainig_note_profile)

#     error_vis.chart_midivel(all_errors_correct_classification, all_training_notes, outout_dir)
#     error_vis.chart_picth(all_errors_correct_classification, all_training_notes, outout_dir)
#     error_vis.chart_pedal(all_errors_correct_classification, all_training_notes, outout_dir)

#     return 0



# def noise_visualization(gt, sigma0, sigma1, sigma2, sigma3, noise_img_gen_dir):
#     """_summary_

#     Args:
#         gt (list): ground truth MIDI roll 
#         sigma0 (list): no noise removal
#         sigma1 (list): sigma for finidng z-score 1
#         sigma2 (list): sigma for finidng z-score 2
#         sigma3 (list): sigma for finidng z-score 3

#     Returns:
#         _type_: _description_
#     """

#     for i, gt_dict in enumerate(gt):

#         gt_img = gt_dict["velocity_roll"][0]
#         sigma0_img = sigma0[i]["velocity_output"][0]
#         sigma1_img = sigma1[i]["velocity_output"][0]
#         sigma2_img = sigma2[i]["velocity_output"][0]
#         sigma3_img = sigma3[i]["velocity_output"][0]

#         #time = np.linspace(0, 20, 621)
#         fig, axs = plt.subplots(1, 5, figsize=(12, 4), gridspec_kw={'wspace': 0.05})

#         axs[0].imshow(gt_img, cmap='hot', aspect='auto', extent=[0, 88, 0, 22])
#         axs[0].set_title('Ground Truth')

#         axs[1].imshow(sigma0_img, cmap='hot', aspect='auto', extent=[0, 88, 0, 20])
#         axs[1].set_title('Raw Output')
#         axs[1].set_yticks([])
#         axs[1].set_yticklabels([])

#         axs[2].imshow(sigma1_img, cmap='hot', aspect='auto', extent=[0, 88, 0, 20])
#         axs[2].set_title('Z-score = 1')
#         axs[2].set_yticks([])
#         axs[2].set_yticklabels([])

#         axs[3].imshow(sigma2_img, cmap='hot', aspect='auto', extent=[0, 88, 0, 20])
#         axs[3].set_title('Z-score = 2')
#         axs[3].set_yticks([])
#         axs[3].set_yticklabels([])

#         axs[4].imshow(sigma3_img, cmap='hot', aspect='auto', extent=[0, 88, 0, 22])
#         axs[4].set_title('Z-score = 3')
#         axs[4].set_yticks([])
#         axs[4].set_yticklabels([])

#         fig.text(0.5, 0.02, '88 keys', ha='center')
#         fig.text(0.08, 0.5, 'Time (seconds)', va='center', rotation='vertical')
#         plt.subplots_adjust(wspace=0.15)
        
#         # Adjust the spacing between subplots
#         plt.tight_layout()
#         file_name = time.strftime("%Y%m%d-%H%M%S") + ".png"
#         plt.savefig(os.path.join(noise_img_gen_dir, file_name), bbox_inches='tight')

#         plt.close(fig)

#     return 0 
