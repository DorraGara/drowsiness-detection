import csv
from fileinput import filename
import os
import os.path
import sys
from subprocess import call
import numpy as np
import cv2



image_dataset_path = '/home/dorra.gara/multi_processed_image_dataset'
rootdir = '/home/dorra.gara/dataset/Training_Evaluation_Dataset'
features = ['mouth', 'eye', 'head', 'drowsiness']
feature_states = {}
data_file_train = []
data_file_test = []
nb_class = {
    'mouth': 3,
    'head': 3,
    'eye': 2,
    'drowsiness': 2
}

def extract_files(frame_length_error):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if (file.endswith(".avi") or file.endswith(".mp4")):
                    video_path = os.path.join(subdir, file)
                    # Get the parts of the file.
                    video_parts = get_video_parts(video_path)

                    train_or_test, filename, glasses, night, baseframename = video_parts

                    get_states(subdir, filename)
                    
                    vidcap = cv2.VideoCapture(video_path)
                    success,image = vidcap.read()
                    frame_count = 0

                    while success: 
                        new_frame_length_error,csv_data = write_frame(video_parts, image, frame_count, frame_length_error)
                        success,image = vidcap.read()
                        frame_count += 1

                        if new_frame_length_error == frame_length_error:
                            if train_or_test == 'train':
                                data_file_train.append(csv_data)
                            else:
                                data_file_test.append(csv_data)

                        frame_length_error = new_frame_length_error

                    print(f"Generated {frame_count} frames for {train_or_test} {baseframename}")
            feature_states.clear()
                    
    path_datafile_train = os.path.join(image_dataset_path, 'data_file_train.csv')
    path_datafile_test = os.path.join(image_dataset_path, 'data_file_test.csv')
    with open(path_datafile_train, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file_train)
    print("Extracted and wrote %d video files TRAIN." % (len(data_file_train)))
    with open(path_datafile_test, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file_test)
    print("Extracted and wrote %d video files TEST." % (len(data_file_test)))
    
    print(f'error length frame: {frame_length_error}')

def write_frame(video_parts, image, frame_count, frame_length_error):

    train_or_test, filename, glasses, night, baseframename = video_parts
    framename = baseframename + str(frame_count) + ".jpg"
    keep_image = True
    for feature in features:
        if len(feature_states[feature]) <= frame_count:
            keep_image = False
            frame_length_error += 1
    csv_data = []
    if keep_image:
        path_image = os.path.join(image_dataset_path, train_or_test, framename)    
        cv2.imwrite(path_image, image)    
        csv_data = [path_image, baseframename, frame_count, glasses, night, feature_states['mouth'][frame_count],feature_states['head'][frame_count],feature_states['eye'][frame_count],feature_states['drowsiness'][frame_count]]        
    
    return frame_length_error,csv_data

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)

    if parts[5] == "Training_Dataset":
        train_or_test = "train"
        filename = parts[8].split(".")[0]
        baseframename = parts[6] + "_" + filename

    else:
        train_or_test = "test"
        filename = parts[7].split(".")[0]
        baseframename = filename 
   
    glasses = 1
    night = 0
    if parts[7].__contains__("noglasses"):
        glasses = 0
    if parts[7].__contains__("night"):
        night = 1
    
    baseframename = baseframename + "_" + str(glasses) + "_" + str(night)

    return train_or_test, filename, glasses, night, baseframename

def get_list(path_file_txt):
    s = open(path_file_txt, 'r').read()
    list_state = np.array(list(s))
    list_state = list_state.astype(np.int32)
    return list_state

def get_states(subdir, filename):
    for file_text in os.listdir(subdir):
        if (file_text.endswith(".txt") and file_text.__contains__(filename)):
            for feature in features:
                if (file_text.__contains__(feature)):
                    path_file_text = os.path.join(subdir, file_text)
                    feature_states[feature] = get_list(path_file_text)
                    break 

def create_folders():
    if not os.path.exists(image_dataset_path):
        os.mkdir(image_dataset_path)
    path_train = os.path.join(image_dataset_path, "train")
    if not os.path.exists(path_train):
        os.mkdir(path_train)
    path_test = os.path.join(image_dataset_path, "test")
    if not os.path.exists(path_test):
        os.mkdir(path_test)
       

def main():
    frame_length_error = 0
    create_folders()
    extract_files(frame_length_error)

if __name__ == '__main__':
    main()