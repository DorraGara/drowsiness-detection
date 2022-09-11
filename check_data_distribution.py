import pandas as pd
import os

image_dataset_path = '/home/dorra.gara/multi_processed_image_dataset'

dataset_eval = pd.read_csv( os.path.join(image_dataset_path,'data_file_evaluate.csv'),
names=["image_path", "image_name", "frame_count", "glasses", "night", "drowsiness" ]
)

dataset_train = pd.read_csv( os.path.join(image_dataset_path,'data_file_train.csv'),
names=["image_path", "image_name", "frame_count", "glasses", "night", "mouth", "head","eye","drowsiness" ]
)

dataset_test = pd.read_csv( os.path.join(image_dataset_path,'data_file_test.csv'),
names=["image_path", "image_name", "frame_count", "glasses", "night", "mouth", "head","eye","drowsiness" ]
)


def check_distribution(dataset):
    columns =dataset.columns

    for col in columns:
        if col in ["glasses", "night", "eye" , "mouth", "head"]:
            dist=dataset[col].value_counts(normalize=True)*100
            print(dist)
            print('-'*30,'\n')



print('='*30,"Train",'='*30)
check_distribution(dataset_train)

print('='*30,"Test",'='*30)
check_distribution(dataset_test)

print('='*30,"Eval",'='*30)
check_distribution(dataset_eval)