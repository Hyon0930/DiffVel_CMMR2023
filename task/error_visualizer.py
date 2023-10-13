import csv
import datetime
import os
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import copy

def chart_simulnotes(scores):
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

    max_simulnote = 0
    for error_profile in scores:
        for error in error_profile:
            if max_simulnote < error["simultaneous_notes"]:
                max_simulnote = error["simultaneous_notes"]

    print("max_simulnote", max_simulnote)
    bucket_size = 5
    num_class = math.ceil(max_simulnote/bucket_size) #TODO:way to make class to be considered
    print("num_class", num_class)

    boxplot_data = []
    boxplot_label = []
    for bucket in range(0,num_class):
        error_bucket = []
        for error_profile in scores:
            for error in error_profile: 
                if bucket*bucket_size < error["simultaneous_notes"] <= (bucket+1)*bucket_size:
                    error_bucket.append(error["note_error"])
        boxplot_label.append("~"+str((bucket+1)*bucket_size))
        boxplot_data.append(error_bucket)

    chart_folder_path = "../inference_result/chart/maestro/"
    simul_label_dict = {"title":"Simultaneous Notes to Error", "x-axis":"Simultaneous Notes", "y-axis":"Estimation Error"}
    chart_png_generator(simul_label_dict, chart_folder_path, boxplot_data, boxplot_label)

    return 0
    
def chart_pedal(scores, all_training_notes, chart_folder_path):
    """_summary_    
    - average error based on pedal vs vel difference (ratio of notes)
        - x: group of 10 pitch, y: error range, (ratio of notes)
    """
    pedal_on = []
    pedal_off = []

    training_pedal_on = 0
    training_pedal_off = 0

    for error_profile in scores:
        for error in error_profile:
            if error["pedal_check"]:
                pedal_on.append(error["note_error"])
            else:
                pedal_off.append(error["note_error"])
            

    boxplot_data = [pedal_on, pedal_off]
    boxplot_label = ["pedal on", "pedal off"]


    for note_profile in all_training_notes:
        for note in note_profile:
            if note["pedal_check"]:
                training_pedal_on += 1
            else:
                training_pedal_off += 1
    print("on, off", training_pedal_on, training_pedal_off)
    ratio_on = training_pedal_on/(training_pedal_on + training_pedal_off) * 100 # format to percentage
    ratio_off = training_pedal_off/(training_pedal_on + training_pedal_off) * 100 # format to percentage
    ratio_list = [ratio_on, ratio_off] 

    

    pedal_label_dict = {"title":"Sustain Pedal Activation", "x-axis":"Sustain Pedal Activation", "y-axis":"Estimation Error"}
    #chart_png_generator(midivel_label_dict, chart_folder_path, boxplot_data, boxplot_label)
    ratio_chart_png_generator(pedal_label_dict, chart_folder_path, boxplot_data, ratio_list, boxplot_label)

    return boxplot_data, boxplot_label


def chart_picth(scores, all_training_notes, chart_folder_path):
    """_summary_
    - average error based on pitch vs vel difference (ratio of notes)
        - x: group of 10 pitch, y: error range, (ratio of notes)
    Args:
        error_profile (_type_): _description_
    """
    pitch = 88

    bucket_size = 10
    num_class = math.ceil(pitch/bucket_size) #TODO:way to make class to be considered
    print("num_class", num_class)

    boxplot_data = []
    boxplot_label = []
    ratio_list = []
    num_data = 0

    for bucket in range(0,num_class):
        error_bucket = []
        for error_profile in scores:
            for error in error_profile:
                if bucket*bucket_size < error["pitch"] <= (bucket+1)*bucket_size:
                    error_bucket.append(error["note_error"])
                    num_data += 1
        boxplot_label.append("~"+str((bucket+1)*bucket_size))
        boxplot_data.append(error_bucket)


    ratio_list = trainingset_ratio(all_training_notes, bucket_size, num_class, "pitch")

    pitch_label_dict = {"title":"Pitch to Error", "x-axis":"Pitch", "y-axis":"Estimation Error"}
    #chart_png_generator(pitch_label_dict, chart_folder_path, boxplot_data, boxplot_label)
    ratio_chart_png_generator(pitch_label_dict, chart_folder_path, boxplot_data, ratio_list, boxplot_label)

    return 0

def chart_midivel(scores, all_training_notes, chart_folder_path):

    midivel = 127

    bucket_size = 10
    num_class = math.ceil(midivel/bucket_size) #TODO:way to make class to be considered
    print("num_class", num_class)

    boxplot_data = []
    boxplot_label = []
    ratio_list = []
    num_data = 0
    
    for bucket in range(0,num_class):
        error_bucket = []
        for error_profile in scores:
            for error in error_profile: 
                if bucket*bucket_size < error["ground truth"] <= (bucket+1)*bucket_size:
                    error_bucket.append(error["note_error"])
                    num_data += 1 
        boxplot_label.append("~"+str((bucket+1)*bucket_size))
        boxplot_data.append(error_bucket)

    assert len(boxplot_data) == len(boxplot_data)
    
    ratio_list = trainingset_ratio(all_training_notes, bucket_size, num_class, "ground truth")



    midivel_label_dict = {"title":"MIDI velocity to Error", "x-axis":"Ground Truth MIDI Velocity", "y-axis":"Estimation Error"}
    #chart_png_generator(midivel_label_dict, chart_folder_path, boxplot_data, boxplot_label)
    ratio_chart_png_generator(midivel_label_dict, chart_folder_path, boxplot_data, ratio_list, boxplot_label)

    return 0


def trainingset_ratio(all_training_notes, bucket_size, num_class, target_key):
    
    ratio_list = []
    lineplot_data = []
    num_data = 0
    
    for bucket in range(0,num_class):
        note_bucket = []
        for note_profile in all_training_notes:
            for note in note_profile: 
                if bucket*bucket_size < note[target_key] <= (bucket+1)*bucket_size:
                    note_bucket.append(note[target_key])
                    num_data += 1 
        lineplot_data.append(note_bucket)

    for bucket in lineplot_data:
        note_ratio = len(bucket)/num_data * 100 # format to percentage
        ratio_list.append(note_ratio)
        print("note_ratio", note_ratio)
    
    return ratio_list



def missclassification_sort(error_profile):
    
    correct_classification = []
    miss_classification = []

    for error in error_profile: 
        if error["classification_check"]:
            correct_classification.append(error)
        else:
            miss_classification.append(error)
    
    assert len(error_profile) == len(correct_classification) + len(miss_classification)

    return correct_classification, miss_classification


def chart_png_generator(label_dict, folder_path, data_list, data_label): 
    title = label_dict["title"].replace(" ", "_")
    file_fullpath = os.path.join(folder_path, title + ".png")
    plt.boxplot(data_list, labels=data_label)
    
    plt.xlabel(label_dict["x-axis"])
    plt.ylabel(label_dict["y-axis"])
    plt.title(label_dict["title"])
    plt.savefig(file_fullpath)
    plt.close()


def ratio_chart_png_generator(label_dict, folder_path, boxdata_list, linedata_list, data_label):
    title = label_dict["title"].replace(" ", "_")
    file_fullpath = os.path.join(folder_path, title + ".png")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.boxplot(boxdata_list, labels=data_label, positions=range(len(boxdata_list)))
    ax2.plot(data_label, linedata_list, 'b-')
    
    ax1.set_xlabel(label_dict["x-axis"])
    ax1.set_ylabel(label_dict["y-axis"])
    ax2.set_ylabel('Note Ratio in Training Set', color='b')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.savefig(file_fullpath, bbox_inches='tight')
    plt.close()


def midivel_check(target_key, error_profile, PNG_PATH, npy_file):

    npy_file, _ = npy_file.split(".")

    # Assuming you have a numpy array of dictionaries called dict_array
    estimations = np.array([item[target_key] for item in error_profile])

    # Define the range and number of bins for the histogram
    bins = np.arange(0, 128, 5)

    # Create the histogram
    plt.hist(estimations, bins=bins)

    # Add labels and show the plot
    plt.xlabel('Estimation')
    plt.ylabel('Frequency')
    plt.title('Histogram of '+ target_key)
    png_filename = npy_file + 'histogram_of_' + target_key + '.png'
    hist_path = os.path.join(PNG_PATH, png_filename)
    plt.savefig(hist_path, dpi=300)
    plt.close()

def pitch_check(error_profile, PNG_PATH, npy_file):

    npy_file, _ = npy_file.split(".")

    # Assuming you have a numpy array of dictionaries called dict_array
    estimations = np.array([item["pitch"] for item in error_profile])

    # Define the range and number of bins for the histogram
    bins = np.arange(0, 88, 1)

    # Create the histogram
    plt.hist(estimations, bins=bins)

    # Add labels and show the plot
    plt.xlabel('Estimation')
    plt.ylabel('Frequency')
    plt.title('Histogram of pitch')
    png_filename = npy_file + 'histogram_of_pitch.png'
    hist_path = os.path.join(PNG_PATH, png_filename)
    plt.savefig(hist_path, dpi=300)
    plt.close()

def remove_gaussian(error_profile, gaussian_removal): 
    
    for note_error in error_profile:

        if note_error["estimation"] < gaussian_removal: 
            note_error["classification_check"] = False
            note_error["estimation"] = 0

        note_error['estimation'] = note_error['estimation'] - gaussian_removal
        note_error['note_error'] = abs(note_error['estimation'] - note_error['ground truth'])

    return error_profile


def check_mean_std(correct_classification): 
    error = []
    for note_error in correct_classification:
        error.append(note_error['note_error'])
    
    mean = np.mean(error)
    std = np.std(error)

    print("mean, std", mean, std)

    return mean, std



def recall_after_gaussian_removal(error_profile):
    
    positive_count = 0

    for note_error in error_profile:
        if note_error["classification_check"]:
            positive_count += 1


    recall = positive_count/len(error_profile)
    print("recall", recall)

    return recall


if __name__ == "__main__":

    PATH_TO_ERROR_NPY = "/home/hk-upf/Documents/workspace/upf/proj/dynamics_tracker/diffvel_inference_result/Diffvel_originalFilm/"
    PNG_PATH = os.path.join(PATH_TO_ERROR_NPY, "png_files/")
    CSV_PATH = os.path.join(PATH_TO_ERROR_NPY, "gaussian_removal_csv/")
    PATH_TO_TRAININGSET_NPY = "/home/hk-upf/Documents/workspace/upf/proj/dynamics_tracker/film_based/inference_result/error_dict/maestro"
    CHART_FOLDER_PATH = "/home/hk-upf/Documents/workspace/upf/proj/dynamics_tracker/diffvel_inference_result/Diffvel_originalFilm/chart"
    error_npy_list = os.listdir(PATH_TO_ERROR_NPY)
    training_npy_list = os.listdir(PATH_TO_TRAININGSET_NPY)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    all_errors_correct_classification = []
    all_training_notes = []
    gaussian_removal = 10



    with open(CSV_PATH+ current_time + "after_gaussian_rm" +str(gaussian_removal)+"film.csv", "w") as result:
        writer=csv.writer(result, delimiter=',' , lineterminator='\n')
        fields = ["name", "mean", "std",  "recall", "gaussian_removal"]
        writer.writerow(fields)
        results = []
        for npy_file in error_npy_list:

            print("npy_file", npy_file)
            if os.path.isdir(os.path.join(PATH_TO_ERROR_NPY, npy_file)):
                # skip directories
                continue
            if npy_file.endswith(".npy"):
                error_profile = np.load(os.path.join(PATH_TO_ERROR_NPY, npy_file), allow_pickle=True)
                # midivel_check("estimation", error_profile, PNG_PATH, npy_file)
                # midivel_check("ground truth", error_profile, PNG_PATH, npy_file)
                # pitch_check(error_profile, PNG_PATH, npy_file)

                error_profile = remove_gaussian(error_profile, gaussian_removal)
                recall = recall_after_gaussian_removal(error_profile)
                correct_classification, miss_classification = missclassification_sort(error_profile)
                all_errors_correct_classification.append(correct_classification)
                mean, std = check_mean_std(correct_classification)
                row = (npy_file, mean, std, recall, gaussian_removal)
                row_for_ave = (mean, std, recall, gaussian_removal)
                writer.writerow(row)
                results.append(row_for_ave)
        
        results_np = np.array(results)
        # Calculate the mean of each column
        mean_results = np.mean(results_np, axis=0)
        row_for_ave = ("ave", mean_results[0], mean_results[1], mean_results[2], mean_results[3])
        writer.writerow(row_for_ave)



    for npy_file in training_npy_list:
        print("npy_file", npy_file)
        trainig_note_profile = np.load(os.path.join(PATH_TO_TRAININGSET_NPY, npy_file), allow_pickle=True)
        all_training_notes.append(trainig_note_profile)


    chart_midivel(all_errors_correct_classification, all_training_notes, CHART_FOLDER_PATH)
    chart_picth(all_errors_correct_classification, all_training_notes, CHART_FOLDER_PATH)
    chart_pedal(all_errors_correct_classification, all_training_notes, CHART_FOLDER_PATH)
    #chart_simulnotes(all_errors_correct_classification)




