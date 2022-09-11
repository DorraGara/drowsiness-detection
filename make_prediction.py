from keras.models import load_model
import os
import xgboost as xgb
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder , LabelBinarizer


output_path = '/home/dorra.gara/training/multi-task-cnn'
image_dataset_path = '/home/dorra.gara/multi_processed_image_dataset'
model_cnn = load_model(os.path.join(output_path,"cnn_model.h5"))

model_xgb = xgb.Booster()
model_xgb.load_model(os.path.join(output_path,"xgb_model.json"))


dataset_eval = pd.read_csv( os.path.join(image_dataset_path,'data_file_evaluate.csv'),
names=["image_path", "image_name", "frame_count", "glasses", "night", "drowsiness" ]
)
# dataset_eval = dataset_eval.sample(n=300)
data_xgb = []


def get_input(path):
    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = tf.image.resize(image_arr,(224,224), method=tf.image.ResizeMethod.BILINEAR).numpy()
    return image_arr/255.

def manual_hot_encoding(data):
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(3))
    hot_encoded = label_binarizer.transform(data)
    return hot_encoded

def reformat_list_prediction(data):

    mouth_encoded = manual_hot_encoding(np.argmax(np.array(data[2]),axis=1))    
    head_encoded = manual_hot_encoding(np.argmax(np.array(data[3]),axis=1))    
    dataset = pd.DataFrame({
        'glasses': np.argmax(np.array(data[0]),axis=1),
        'night': np.argmax(np.array(data[1]),axis=1),
        'eye': np.argmax(np.array(data[4]),axis=1),
        'mouth_0': mouth_encoded[:,0],
        'mouth_1': mouth_encoded[:,1],
        'mouth_2': mouth_encoded[:,2],
        'head_0': head_encoded[:,0],
        'head_1': head_encoded[:,1],
        'head_2': head_encoded[:,2],
    })
    return dataset


dataset_eval = dataset_eval.drop('image_path',axis=1)
dataset_eval = dataset_eval.drop('frame_count',axis=1)
dataset_eval = dataset_eval.drop('night',axis=1)
dataset_eval = dataset_eval.drop('glasses',axis=1)

eval_datagen = ImageDataGenerator(
            rescale=1 / 255
            )

eval_datagen_flow = eval_datagen.flow_from_dataframe(
    dataframe = dataset_eval,
    directory=os.path.join(image_dataset_path,'evaluate'),
    x_col="image_name", 
    y_col="drowsiness",
    #we are doing regression, so we will assign class_mode to 'raw'
    class_mode="raw",
    target_size=(224,224),
    batch_size=32,
    seed=12345,
    )


prediction_cnn = model_cnn.predict(eval_datagen_flow)

dataset_xgb = reformat_list_prediction(prediction_cnn)
print(dataset_xgb)

dtest = xgb.DMatrix(
	dataset_xgb,
	label=dataset_eval["drowsiness"]
)
predictions = model_xgb.predict(dtest)
predictions = [round(value) for value in predictions]
accuracy = accuracy_score(dataset_eval["drowsiness"], predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))