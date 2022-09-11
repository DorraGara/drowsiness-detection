import csv
from fileinput import filename
import os
import os.path
import sys
from subprocess import call
import numpy as np
import cv2



image_dataset_path = '/home/dorra.gara/multi_processed_image_dataset'
rootdir = '/home/dorra.gara/dataset/Testing_Dataset'
path_text_states = os.path.join(rootdir,"test_label_txt","wh")

data_file_test = []


def extract_files(frame_length_error):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if (file.endswith(".mp4")):
                    video_path = os.path.join(subdir, file)
                    # Get the parts of the file.
                    video_parts = get_video_parts(video_path)

                    filename, glasses, night = video_parts
                    states = get_states(filename)
                    
                    vidcap = cv2.VideoCapture(video_path)
                    success,image = vidcap.read()
                    frame_count = 0

                    while success: 
                        new_frame_length_error,csv_data = write_frame(video_parts, image, frame_count, frame_length_error,states)
                        success,image = vidcap.read()
                        frame_count += 1

                        if new_frame_length_error == frame_length_error:
                            data_file_test.append(csv_data)

                        frame_length_error = new_frame_length_error

                    print(f"Generated {frame_count} frames for {filename}")
                    
    path_datafile_test = os.path.join(image_dataset_path, 'data_file_evaluate.csv')
    with open(path_datafile_test, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file_test)
    print("Extracted and wrote %d video files TEST." % (len(data_file_test)))
    
    print(f'error length frame: {frame_length_error}')

def write_frame(video_parts, image, frame_count, frame_length_error, states):

    filename, glasses, night, = video_parts
    framename = filename + "_" + str(frame_count) + ".jpg"
    keep_image = True
    if len(states) <= frame_count:
        keep_image = False
        frame_length_error += 1
    csv_data = []
    if keep_image:
        path_image = os.path.join(image_dataset_path, "evaluate", framename)    
        cv2.imwrite(path_image, image)    
        csv_data = [path_image, framename, frame_count, glasses, night, states[frame_count]]
    
    return frame_length_error,csv_data

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)

    filename = parts[5].split(".")[0]

    glasses = 1
    night = 0
    if parts[5].__contains__("noglasses"):
        glasses = 0
    if parts[5].__contains__("night"):
        night = 1
    
    return filename, glasses, night

def get_list(path_file_txt):
    s = open(path_file_txt, 'r').read()
    list_s = list(s)
    if(list_s[-1] == '\n'):
        list_s.pop()
    list_state = np.array(list_s)
    list_state = list_state.astype(np.int32)
    return list_state

def get_states(filename):
    states = []
    for file_text in os.listdir(path_text_states):
        if (file_text.endswith(".txt") and file_text.__contains__(filename)):
            path_file_text = os.path.join(path_text_states, file_text)
            states = get_list(path_file_text)
    return states

def create_folders():
    path_test = os.path.join(image_dataset_path, "evaluate")
    if not os.path.exists(path_test):
        os.mkdir(path_test)
       

def main():
    frame_length_error = 0
    create_folders()
    extract_files(frame_length_error)

if __name__ == '__main__':
    main()