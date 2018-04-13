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
import keras
import tensorflow as tf
import time
import gc
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class mdl_ensemble(object):

    def __init__(self):
        if not os.path.exists('models/ensemble1'):
            os.makedirs('models/ensemble1')

        if not os.path.exists('submissions/ensemble1'):
            os.makedirs('submissions/ensemble1')

        if not os.path.exists('weights/ensemble1'):
            os.makedirs('weights/ensemble1')

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
        X_normalized = self.normalize_scale(X)

        return X_normalized

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
        most_proba.append(int(maxi_1_value[0]))
        siguientes_2 = test_preds_vector[np.where(test_preds_vector != maxi_1)[0]]

        maxi_2 = max(siguientes_2)
        maxi_2_value = np.where(test_preds_vector == maxi_2)[0]
        most_proba.append(int(maxi_2_value[0]))
        siguientes_3 = siguientes_2[np.where(siguientes_2 != maxi_2)[0]]

        maxi_3 = max(siguientes_3)
        maxi_3_value = np.where(test_preds_vector == maxi_3)[0]
        most_proba.append(int(maxi_3_value[0]))
        siguientes_4 = siguientes_3[np.where(siguientes_3 != maxi_3)[0]]

        maxi_4 = max(siguientes_4)
        maxi_4_value = np.where(test_preds_vector == maxi_4)[0]
        most_proba.append(int(maxi_4_value[0]))

        most_proba = np.array(most_proba)

        return most_proba

    def submit(self, name, test_preds, test_images):
        with open('submissions/ensemble1/{}.csv'.format(name), 'w') as f:
            with warnings.catch_warnings():
                f.write("Image,Id\n")
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                n = 0
                for image in test_images:
                    n += 1
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
        most_proba.append(int(maxi_2_value[0]))
        siguientes_3 = siguientes_2[np.where(siguientes_2 != maxi_2)[0]]

        maxi_3 = max(siguientes_3)
        maxi_3_value = np.where(test_preds_vector == maxi_3)[0]
        most_proba.append(int(maxi_3_value[0]))
        siguientes_4 = siguientes_3[np.where(siguientes_3 != maxi_3)[0]]

        maxi_4 = max(siguientes_4)
        maxi_4_value = np.where(test_preds_vector == maxi_4)[0]
        most_proba.append(int(maxi_4_value[0]))
        siguientes_5 = siguientes_4[np.where(siguientes_4 != maxi_4)[0]]

        maxi_5 = max(siguientes_5)
        maxi_5_value = np.where(test_preds_vector == maxi_5)[0]
        most_proba.append(int(maxi_5_value[0]))

        most_proba = np.array(most_proba)

        return most_proba

    def new_submit(self,name, test_preds, test_images, thr=0.5):
        with open('submissions/ensemble1/{}.csv'.format(name), 'w') as f:
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
    mdl = mdl_ensemble()

    # Load data
    print('Loading data...')
    train = pd.read_csv('data/train_ensemble_1.csv')
    X = train.loc[:, train.columns != 'Id']
    n_cols = len(X.columns)
    y = train['Id']
    test_images = mdl.data_train_extraction('data/test')

    # Label encoder y one hot. Necessary for keras models
    y_label = mdl.label_encoder.fit_transform(y)
    y_one_hot = mdl.label_binarizer.fit_transform(y_label)

    del train
    gc.collect()

    t0 = time.time()
    n_output = len(y.unique())

    # NAMES #
    model_name = 'model_ensemble1_7'
    submit_name = 'submission_ensemble7'
    #########
    print('Generating model {}...'.format(model_name))
    model = Sequential()

    model.add(Dense(4, activation='relu', input_dim=n_cols))
    model.add(Dense(8, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(2048, activation='relu'))
    model.add(Dense(n_cols, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    earlyStopping = keras.callbacks.EarlyStopping(monitor='acc', patience=5, verbose=1, mode='auto')
    print('Training model...')
    # skf = StratifiedKFold(n_splits=10, shuffle=True)
    #
    # # Loop through the indices the split() method returns
    # for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
    #     print("Training on fold " + str(index + 1) + "/10...")
    #     # Generate batches from indices
    #     xtrain, xval = X[train_indices], X[val_indices]
    #     ytrain, yval = y_one_hot[train_indices], y_one_hot[val_indices]
    #
    #     history = model.fit(xtrain,
    #                         ytrain,
    #                         epochs=200,
    #                         verbose=1,  # 2 -> Only 1 print per epoch
    #                         callbacks=[earlyStopping],
    #                         batch_size=2,
    #                         validation_data=(xval, yval)
    #                         )
    #     accuracy_history = history.history['acc']
    #     val_accuracy_history = history.history['val_acc']
    #     print("Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: "
    #           + str(val_accuracy_history[-1]))
    history = model.fit(X,
                        y_one_hot,
                        epochs=200,
                        verbose=2, # 2 -> Only 1 print per epoch
                        callbacks=[earlyStopping],
                        batch_size=64
                        )

    model.save_weights('weights/ensemble1/{}.h5'.format(model_name))
    model.save('models/ensemble1/{}.h5'.format(model_name))

    print('Training time: {} min'.format(round((time.time() - t0) / 60, 2)))

    del X
    gc.collect()
    print('Predictions...')
    # from keras.models import load_model
    # model = load_model('models/ensemble1/{}.h5'.format(model_name))
    # model.load_weights('weights/ensemble1/{}.h5'.format(model_name))
    X_test = pd.read_csv('data/test_ensemble_1.csv')
    test_preds = model.predict_proba(X_test)
    del X_test
    gc.collect()

    t0 = time.time()
    mdl.new_submit(submit_name, test_preds, test_images, thr=0.99)
    mdl.submit(submit_name+'_thr099', test_preds, test_images)
    print('Submit prediction in: {} secs'.format(round(time.time() - t0, 1)))