import numpy as np
import keras
import tensorflow as tf

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size,
                 input_size=(224, 224, 3),
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_glasses = df[y_col['glasses']].nunique()
        self.n_night = df[y_col['night']].nunique()
        self.n_mouth = df[y_col['mouth']].nunique()
        self.n_head = df[y_col['head']].nunique()
        self.n_eye = df[y_col['eye']].nunique()

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, target_size):
    
        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        
        glasses_batch = batches[self.y_col['glasses']]
        night_batch = batches[self.y_col['night']]
        mouth_batch = batches[self.y_col['mouth']]
        head_batch = batches[self.y_col['head']]
        eye_batch = batches[self.y_col['eye']]
        
        X_batch = np.asarray([self.__get_input(x, self.input_size) for x in path_batch])

        y0_batch = np.asarray([self.__get_output(y, self.n_glasses) for y in glasses_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_night) for y in night_batch])
        y2_batch = np.asarray([self.__get_output(y, self.n_mouth) for y in mouth_batch])
        y3_batch = np.asarray([self.__get_output(y, self.n_head) for y in head_batch])
        y4_batch = np.asarray([self.__get_output(y, self.n_eye) for y in eye_batch])

        return X_batch, tuple([y0_batch, y1_batch,y2_batch,y3_batch,y4_batch])
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size