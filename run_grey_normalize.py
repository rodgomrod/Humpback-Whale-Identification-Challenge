import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
import pandas as pd
from scipy import misc
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import cv2
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import keras
import tensorflow as tf
import time
import gc
from sklearn.utils import class_weight

class mdl_normalize(object):

    def __init__(self):
        if not os.path.exists('models/grey_normalize/'):
            os.makedirs('models/grey_normalize/')

        if not os.path.exists('submissions/grey_normalize'):
            os.makedirs('submissions/grey_normalize/')

        if not os.path.exists('weights/grey_normalize'):
            os.makedirs('weights/grey_normalize/')

        self.label_encoder = LabelEncoder()
        self.label_binarizer = LabelBinarizer()

    def data_train_extraction(self, folder_train):
        imagepaths = [x for x in os.walk(folder_train)][0][-1]
        return imagepaths

    def resize_image(self, path, resolution):
        img = cv2.imread(path)
        (b, g, r) = cv2.split(img)
        img = cv2.merge([r, g, b])
        image = misc.imresize(img, (resolution, resolution), mode=None)
        return image

    def normalize_scale(self, image_data):
        a = -0.5
        b = 0.5
        scale_min = 0
        scale_max = 255
        return a + (((image_data - scale_min) * (b - a)) / (scale_max - scale_min))

    def normalize_images(self, X):
        X_grey = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in X]
        X_grey = np.asarray(X_grey)
        X_normalized_grey = X_grey

        # X_normalized_grey = self.normalize_scale(X_grey)

        # X_normalized_grey = [cv2.GaussianBlur(i, (5, 5), 0) for i in X] #blurred
        # X_normalized_grey = np.asarray(X_normalized_grey)

        return X_normalized_grey

    def predict_5_better(self, test_preds_vector):
        most_proba = list()
        maxi_1 = max(test_preds_vector)
        maxi_1_value = np.where(test_preds_vector == maxi_1)[0]
        most_proba.append(int(maxi_1_value))
        siguientes_2 = test_preds_vector[np.where(test_preds_vector != maxi_1)[0]]

        maxi_2 = max(siguientes_2)
        maxi_2_value = np.where(test_preds_vector == maxi_2)[0]
        most_proba.append(int(maxi_2_value))
        siguientes_3 = siguientes_2[np.where(siguientes_2 != maxi_2)[0]]

        maxi_3 = max(siguientes_3)
        maxi_3_value = np.where(test_preds_vector == maxi_3)[0]
        most_proba.append(int(maxi_3_value))
        siguientes_4 = siguientes_3[np.where(siguientes_3 != maxi_3)[0]]

        maxi_4 = max(siguientes_4)
        maxi_4_value = np.where(test_preds_vector == maxi_4)[0]
        most_proba.append(int(maxi_4_value))
        siguientes_5 = siguientes_4[np.where(siguientes_4 != maxi_4)[0]]

        maxi_5 = max(siguientes_5)
        maxi_5_value = np.where(test_preds_vector == maxi_5)[0]
        most_proba.append(int(maxi_5_value))

        most_proba = np.array(most_proba)

        return most_proba

    def predict_4_better(self, test_preds_vector):
        most_proba = list()
        maxi_1 = max(test_preds_vector)
        maxi_1_value = np.where(test_preds_vector == maxi_1)[0]
        # print(maxi_1_value)
        most_proba.append(int(maxi_1_value[0]))
        siguientes_2 = test_preds_vector[np.where(test_preds_vector != maxi_1)[0]]

        maxi_2 = max(siguientes_2)
        maxi_2_value = np.where(test_preds_vector == maxi_2)[0]
        most_proba.append(int(maxi_2_value))
        siguientes_3 = siguientes_2[np.where(siguientes_2 != maxi_2)[0]]

        maxi_3 = max(siguientes_3)
        maxi_3_value = np.where(test_preds_vector == maxi_3)[0]
        most_proba.append(int(maxi_3_value))
        siguientes_4 = siguientes_3[np.where(siguientes_3 != maxi_3)[0]]

        maxi_4 = max(siguientes_4)
        maxi_4_value = np.where(test_preds_vector == maxi_4)[0]
        most_proba.append(int(maxi_4_value[0]))

        most_proba = np.array(most_proba)

        return most_proba

    def submit(self, name, test_preds, test_images):
        with open('submissions/grey_normalize/{}.csv'.format(name), 'w') as f:
            with warnings.catch_warnings():
                f.write("Image,Id\n")
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                n = 0
                for image in test_images:
                    n += 1
                    # PREDICTIONS WITHOUT NEW_WHALE:
                    # label_arg = self.predict_5_better(test_preds[n - 1])
                    # preds = self.label_encoder.inverse_transform(label_arg)
                    # preds = preds.tolist()
                    # PREDICTIONS WITH NEW_WHALE:
                    label_arg = self.predict_4_better(test_preds[n - 1])
                    preds = self.label_encoder.inverse_transform(label_arg)
                    preds = ['new_whale'] + preds.tolist()
                    predicted_tags = " ".join(preds)
                    f.write("%s,%s\n" % (image, predicted_tags))
        f.close()

    def predict_new_5_better(self, test_preds_vector, thr=0.5):
        most_proba = list()
        maxi_1 = max(test_preds_vector)
        if maxi_1 < thr:
            most_proba.append('new_whale')
        else:
            maxi_1_value = np.where(test_preds_vector == maxi_1)[0]
            most_proba.append(int(maxi_1_value[0]))
        siguientes_2 = test_preds_vector[np.where(test_preds_vector != maxi_1)[0]]

        maxi_2 = max(siguientes_2)
        maxi_2_value = np.where(test_preds_vector == maxi_2)[0]
        most_proba.append(int(maxi_2_value))
        siguientes_3 = siguientes_2[np.where(siguientes_2 != maxi_2)[0]]

        maxi_3 = max(siguientes_3)
        maxi_3_value = np.where(test_preds_vector == maxi_3)[0]
        most_proba.append(int(maxi_3_value))
        siguientes_4 = siguientes_3[np.where(siguientes_3 != maxi_3)[0]]

        maxi_4 = max(siguientes_4)
        maxi_4_value = np.where(test_preds_vector == maxi_4)[0]
        most_proba.append(int(maxi_4_value[0]))
        siguientes_5 = siguientes_4[np.where(siguientes_4 != maxi_4)[0]]

        maxi_5 = max(siguientes_5)
        maxi_5_value = np.where(test_preds_vector == maxi_5)[0]
        most_proba.append(int(maxi_5_value))

        most_proba = np.array(most_proba)

        return most_proba

    def new_submit(self,name, test_preds, test_images, thr=0.5):
        with open('submissions/grey_normalize/{}.csv'.format(name), 'w') as f:
            with warnings.catch_warnings():
                f.write("Image,Id\n")
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                n = 0
                for image in test_images:
                    n += 1
                    label_arg = self.predict_new_5_better(test_preds[n - 1], thr)
                    if label_arg[0] == 'new_whale':
                        preds = self.label_encoder.inverse_transform(list(map(int, label_arg[1:])))
                        preds = ['new_whale'] + preds.tolist()
                    else:
                        preds = self.label_encoder.inverse_transform(label_arg)
                    predicted_tags = " ".join(preds)
                    f.write("%s,%s\n" % (image, predicted_tags))
        f.close()


    def load_Xdata(self, directory):
        X = np.load(directory)['arr_0']

        return X

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    mdl = mdl_normalize()

    # Load data (train) saved in array format

    print('Starting data load and processing (normalize)...')
    input_size = 56 # 56, 128 or 200
    X_train = mdl.load_Xdata('data/array_data/X_train_{}.npz'.format(input_size))

    # extract y labels sorted
    Ids = pd.read_csv('data/train.csv')
    ImageToLabelDict = dict(zip(Ids['Image'], Ids['Id']))
    sorted_image = list()
    train_images = mdl.data_train_extraction('data/train')
    test_images = mdl.data_train_extraction('data/test')
    for img in train_images:
        sorted_image.append(ImageToLabelDict[img])
    df_train = pd.DataFrame({'Image': train_images, 'Id': sorted_image})
    # Delete images with labels 'new_whale'
    df_train = df_train[df_train['Id'] != 'new_whale']

    # Select only 1 per class
    # df_train = df_train.reset_index(drop=True)
    # df_train.drop_duplicates(subset='Id', keep='first', inplace=True)
    # X_train = X_train[df_train.index]

    # Select Y labels to train
    y = df_train['Id']

    # Label encoder y one hot. Necessary for keras models
    y_label = mdl.label_encoder.fit_transform(y)
    y_one_hot = mdl.label_binarizer.fit_transform(y_label)

    # Normalize images
    X_normalized = mdl.normalize_images(X_train)
    X_normalized = X_normalized.reshape(X_normalized.shape[0], 1, 56, 56)
    del X_train
    gc.collect()

    # Balancing data
    # class_weight = class_weight.compute_class_weight('balanced',
    #                                                  np.unique(y_label),
    #                                                  y_label)


    t0 = time.time()
    n_output = len(df_train['Id'].unique())

    # NAMES #
    model_name = 'model_56_11'
    submit_name = 'submission_56_11'
    #########

    print('Generating model {}...'.format(model_name))
    model = Sequential()


    # model.add(Convolution2D(4, 1, 1, input_shape=(1, input_size, input_size), activation="relu"))
    # # model.add(MaxPooling2D((2, 2)))
    # # model.add(Dropout(0.2))
    # model.add(Convolution2D(8, 1, 1, activation="relu"))
    # # model.add(MaxPooling2D((2, 2)))
    # # model.add(Dropout(0.2))
    # # model.add(Convolution2D(64, 2, 2, activation="sigmoid"))
    # # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())
    # model.add(Dense(n_output, activation="softmax"))

    model.add(Convolution2D(8, 1, 1, input_shape=(1, input_size, input_size), activation="relu"))
    model.add(Activation('relu'))
    model.add(Convolution2D(8, 1, 1))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Convolution2D(16, 1, 1))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 1, 1))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Convolution2D(64, 1, 1))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 1, 1))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Convolution2D(128, 1, 1))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 1, 1))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Flatten())
    model.add(Dense(n_output, activation="softmax"))

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    earlyStopping = keras.callbacks.EarlyStopping(monitor='acc', patience=5, verbose=1, mode='auto')

    # print('Pre-processing images')
    # gen = ImageDataGenerator(
    #     rotation_range=360.,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     zoom_range=0.5,
    #     horizontal_flip=True,
    #     vertical_flip=True
    # )
    # model.fit_generator(gen.flow(X_normalized, y_one_hot),
    #                     steps_per_epoch=100,
    #                     epochs=200,
    #                     verbose=2,
    #                     shuffle=True,
    #                     callbacks=[earlyStopping])

    print('Training model...')
    history = model.fit(X_normalized,
                        y_one_hot,
                        epochs=200,
                        verbose=2, # 2 -> Only 1 print per epoch
                        callbacks=[earlyStopping],
                        # class_weight=class_weight,
                        batch_size = 512,
                        # validation_split = 0.15,
                        )

    model.save_weights('weights/grey_normalize/{}.h5'.format(model_name))
    model.save('models/grey_normalize/{}.h5'.format(model_name))

    print('Training time: {} min'.format(round((time.time() - t0) / 60, 2)))

    del X_normalized
    gc.collect()
    print('Predictions...')
    # model = load_model('models/grey_normalize/model_whitout_new_whale_56_4_balanced.h5')
    # model.load_weights('weights/grey_normalize/model_whitout_new_whale_56_4_balanced.h5')
    X_test = mdl.load_Xdata('data/array_data/X_test_{}.npz'.format(input_size))
    test_normalized = mdl.normalize_images(X_test)
    test_normalized = test_normalized.reshape(test_normalized.shape[0], 1, 56, 56)
    del X_test
    gc.collect()
    test_preds = model.predict_proba(test_normalized)
    del test_normalized
    gc.collect()

    t0 = time.time()
    mdl.submit(submit_name, test_preds, test_images)
    # mdl.new_submit(submit_name+'_thr098', test_preds, test_images, thr=0.98)
    print('Submit prediction in: {} secs'.format(round(time.time() - t0, 1)))