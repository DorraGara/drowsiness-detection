from gc import callbacks
from pyexpat import model
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import keras.layers as tfl
from keras.applications.mobilenet_v2 import MobileNetV2
from data_generator import CustomDataGen
from xgboost import XGBClassifier
import csv
from keras.optimizers import adam_v2
from keras.metrics import Precision, Recall
from tabulate import tabulate

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)


image_dataset_path = '/home/dorra.gara/multi_processed_image_dataset'
output_path = '/home/dorra.gara/training/multi-task-cnn'

dataset_train = pd.read_csv( os.path.join(image_dataset_path,'data_file_train.csv'),
names=["image_path", "image_name", "frame_count", "glasses", "night", "mouth", "head","eye","drowsiness" ]
)

dataset_test = pd.read_csv( os.path.join(image_dataset_path,'data_file_test.csv'),
names=["image_path", "image_name", "frame_count", "glasses", "night", "mouth", "head","eye","drowsiness" ]
)

print(dataset_train[:5])

def data_augmenter():
    data_augmentation = tf.keras.Sequential([
        tfl.RandomFlip("horizontal"),
        tfl.RandomRotation(0.2),
    ])
    
    return data_augmentation

data_augmentation = data_augmenter()
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def build_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):    
    input_shape = image_shape + (3,)
    
    base_model = MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)      
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x = tfl.Dense(1024,activation='relu')(x) #dense layer 2
    x = tfl.Dense(512,activation='relu')(x) #dense layer 3

    prediction_layer_glasses  = tf.keras.layers.Dense(2, activation='softmax', name="glasses")
    prediction_layer_night  = tf.keras.layers.Dense(2, activation='softmax', name="night")
    prediction_layer_mouth  = tf.keras.layers.Dense(3, activation='softmax', name="mouth")
    prediction_layer_head  = tf.keras.layers.Dense(3, activation='softmax', name="head")
    prediction_layer_eye  = tf.keras.layers.Dense(2, activation='softmax', name="eye")

    outputs_glasses = prediction_layer_glasses(x) 
    outputs_night = prediction_layer_night(x) 
    outputs_mouth = prediction_layer_mouth(x) 
    outputs_head = prediction_layer_head(x) 
    outputs_eye = prediction_layer_eye(x) 


    model = tf.keras.Model(inputs, outputs=[outputs_glasses,outputs_night,outputs_mouth,outputs_head,outputs_eye])
    
    return model


opt = adam_v2.Adam(learning_rate=0.0001)
model_cnn = build_model()
model_cnn.compile(
    loss={
        'glasses': 'categorical_crossentropy',
        'night': 'categorical_crossentropy',
        'mouth': 'categorical_crossentropy',
        'head': 'categorical_crossentropy',
        'eye': 'categorical_crossentropy',
    },
    optimizer=opt,
    metrics= [Precision(), Recall()]
)
model_cnn.summary()

def get_f1(precision,recall):
    if ((precision + recall) == 0):
        return None
    else:
        return (2 * precision * recall) / (precision + recall)

class Logger(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        glasses_loss = logs.get('glasses_loss')
        night_loss = logs.get('night_loss')
        mouth_loss = logs.get('mouth_loss')
        head_loss = logs.get('head_loss')
        eye_loss = logs.get('eye_loss')
        val_glasses_loss = logs.get('val_glasses_loss')
        val_night_loss = logs.get('val_night_loss')
        val_mouth_loss = logs.get('val_mouth_loss')
        val_head_loss = logs.get('val_head_loss')
        val_eye_loss = logs.get('val_eye_loss')


        glasses_precision = logs.get('glasses_precision')
        night_precision = logs.get('night_precision')
        mouth_precision = logs.get('mouth_precision')
        head_precision = logs.get('head_precision')
        eye_precision = logs.get('eye_precision')
        val_glasses_precision = logs.get('val_glasses_precision')
        val_night_precision = logs.get('val_night_precision')
        val_mouth_precision = logs.get('val_mouth_precision')
        val_head_precision = logs.get('val_head_precision')
        val_eye_precision = logs.get('val_eye_precision')

        glasses_recall = logs.get('glasses_recall')
        night_recall = logs.get('night_recall')
        mouth_recall = logs.get('mouth_recall')
        head_recall = logs.get('head_recall')
        eye_recall = logs.get('eye_recall')
        val_glasses_recall = logs.get('val_glasses_recall')
        val_night_recall = logs.get('val_night_recall')
        val_mouth_recall = logs.get('val_mouth_recall')
        val_head_recall = logs.get('val_head_recall')
        val_eye_recall = logs.get('val_eye_recall')


        print('='*20,epoch+1,'='*20)
        metrics_results = [
            ['glasses',glasses_loss, glasses_precision, glasses_recall, get_f1(glasses_precision,glasses_recall)],
            ['night',night_loss, night_precision, night_recall, get_f1(night_precision,night_recall)],
            ['mouth',mouth_loss, mouth_precision, mouth_recall, get_f1(mouth_precision,mouth_recall)],
            ['head',head_loss, head_precision, head_recall, get_f1(head_precision,head_recall)],
            ['eye',eye_loss, eye_precision, eye_recall, get_f1(eye_precision,eye_recall)],
            ['val_glasses',val_glasses_loss, val_glasses_precision, val_glasses_recall, get_f1(val_glasses_precision,val_glasses_recall)],
            ['val_night',val_night_loss, val_night_precision, val_night_recall, get_f1(val_night_precision,val_night_recall)],
            ['val_mouth',val_mouth_loss, val_mouth_precision, val_mouth_recall, get_f1(val_mouth_precision,val_mouth_recall)],
            ['val_head',val_head_loss, val_head_precision, val_head_recall, get_f1(val_head_precision,val_head_recall)],
            ['val_eye',val_eye_loss, val_eye_precision, val_eye_recall, get_f1(val_eye_precision,val_eye_recall)],
        ]
        print (tabulate(metrics_results, headers=['Feature','Loss','Precision','Recall','F1-score']))
        with open(os.path.join(output_path,"logs_cnn_f1.txt"), 'w') as f:
            epoch_header = '==============================' + str(epoch+1) + '=============================='
            f.write(epoch_header)
            f.write(tabulate(metrics_results, headers=['Feature','Loss','Precision','Recall','F1-score']))
            f.write('\n')
        print('CNN logs saved for epoch:',epoch+1,'.')


train_generator = CustomDataGen(dataset_train,
                        X_col={'path':'image_path'},
                        y_col={'glasses': 'glasses', 'night': 'night','mouth':'mouth','head':'head','eye':'eye'},
                        batch_size=BATCH_SIZE, input_size=IMG_SIZE)

val_generator = CustomDataGen(dataset_test,
                        X_col={'path':'image_path'},
                        y_col={'glasses': 'glasses', 'night': 'night','mouth':'mouth','head':'head','eye':'eye'},
                        batch_size=BATCH_SIZE, input_size=IMG_SIZE)
history = model_cnn.fit(
        train_generator,
        validation_data=val_generator,
        epochs = 10,
        steps_per_epoch = 30,
        validation_steps = 10,
        # steps_per_epoch = dataset_train.shape[0] / BATCH_SIZE,
        # validation_steps= dataset_test.shape[0] / BATCH_SIZE,
        callbacks = [
            Logger(),            
        ],
        verbose=False
)


# logs_csv_path = os.path.join(output_path,"logs_cnn_f1.txt")
# serialize model to HDF5
model_cnn.save(os.path.join(output_path,"cnn_model_f1.h5"))
print('CNN model saved.')

# with open(logs_csv_path, 'w') as fout:
#     for line in logs_csv:
#         fout.write(line)
#         fout.write('\n')

def prepare_data_xgb(dataset_train,dataset_test):
    dataset_train_tabular = dataset_train
    del dataset_train_tabular["frame_count"]
    del dataset_train_tabular["image_path"]
    del dataset_train_tabular["image_name"]

    X_train = dataset_train_tabular.iloc[:, 0:5]
    X_train_encoded = pd.get_dummies(X_train, columns=['mouth','head'])
    X_train_encoded = X_train_encoded.values
    y_train = dataset_train_tabular.iloc[:, 5].values
    
    print(X_train_encoded[:5])
    print(y_train[:5])

    dataset_test_tabular = dataset_test
    del dataset_test_tabular["frame_count"]
    del dataset_test_tabular["image_path"]
    del dataset_test_tabular["image_name"]

    X_test = dataset_test_tabular.iloc[:, 0:5]
    X_test_encoded = pd.get_dummies(X_test, columns=['mouth','head'])
    X_test_encoded = X_test_encoded.values
    y_test = dataset_test_tabular.iloc[:, 5].values

    return X_train_encoded, y_train, X_test_encoded,y_test

def xgb_model(X_train_encoded, y_train, X_test_encoded,y_test):

    params = {
		'max_depth':12,
		'eta':0.05,
		'objective':'binary:logistic',
		'early_stopping_rounds':10,
		'eval_metric':'aucpr'
	}

    
    model = XGBClassifier(**params)
    model = model.fit( X_train_encoded,y_train,eval_set=[(X_test_encoded,y_test)])
    
    predictions = model.predict(X_test_encoded)
    print(predictions)
    print(y_test)

    return model

X_train_encoded, y_train, X_test_encoded,y_test = prepare_data_xgb(dataset_train,dataset_test)
xgb_model_classifier = xgb_model(X_train_encoded, y_train, X_test_encoded,y_test)
xgb_model_classifier.save_model(os.path.join(output_path,"xgb_model.json"))
print('Xgboost model saved.')