import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import misc
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import cv2
import h5py
import time
import gc
from keras.models import load_model

class ensemble_dataset(object):

    def __init__(self):
        # if not os.path.exists('models/ensemble/'):
        #     os.makedirs('models/ensemble/')
        #
        # if not os.path.exists('submissions/ensemble'):
        #     os.makedirs('submissions/ensemble/')
        #
        # if not os.path.exists('weights/ensemble'):
        #     os.makedirs('weights/ensemble/')

        self.label_encoder = LabelEncoder()
        self.label_binarizer = LabelBinarizer()

    def load_m(self, pre_type, model_name):
        print('Loading {} model'.format(model_name))
        model = load_model('models/{0}/{1}.h5'.format(pre_type, model_name))
        model.load_weights('weights/{0}/{1}.h5'.format(pre_type, model_name))
        return model

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

    def grey_normalize_images(self, X):
        X_grey = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in X]
        X_grey = np.asarray(X_grey)
        X_normalized_grey = self.normalize_scale(X_grey)

        return X_normalized_grey

    def normalize_images(self, X):
        X_normalized = self.normalize_scale(X)

        return X_normalized

    def load_Xdata(self, directory):
        X = np.load(directory)['arr_0']

        return X

if __name__ == '__main__':
    t0 = time.time()
    train_ensemble_1 = pd.DataFrame()
    test_ensemble_1 = pd.DataFrame()

    ed = ensemble_dataset()

    # Loading normalized and grey_normalized models
    model_n_1 = ed.load_m('normalize', 'model_whitout_new_whale_56_1')
    model_n_2 = ed.load_m('normalize', 'model_whitout_new_whale_56_2')
    model_n_3 = ed.load_m('normalize', 'model_whitout_new_whale_56_3')
    model_n_4 = ed.load_m('normalize', 'model_whitout_new_whale_56_4')
    model_n_5 = ed.load_m('normalize', 'model_whitout_new_whale_56_5')
    model_n_6 = ed.load_m('normalize', 'model_whitout_new_whale_56_6')
    model_n_7 = ed.load_m('normalize', 'model_whitout_new_whale_56_7_balanced')

    normalized_models = [
        model_n_1,
        model_n_2,
        model_n_3,
        model_n_4,
        model_n_5,
        model_n_6,
        model_n_7
    ]

    model_gn_1 = ed.load_m('grey_normalize', 'model_whitout_new_whale_56_1_balanced')
    model_gn_2 = ed.load_m('grey_normalize', 'model_whitout_new_whale_56_2_balanced')
    model_gn_3 = ed.load_m('grey_normalize', 'model_whitout_new_whale_56_3_balanced')
    model_gn_4 = ed.load_m('grey_normalize', 'model_whitout_new_whale_56_4_balanced')
    model_gn_5 = ed.load_m('grey_normalize', 'model_whitout_new_whale_56_5_balanced')
    model_gn_6 = ed.load_m('grey_normalize', 'model_whitout_new_whale_56_6_balanced')

    grey_normalized_models = [
        model_gn_1,
        model_gn_2,
        model_gn_3,
        model_gn_4,
        model_gn_5,
        model_gn_6
    ]

    # Load data (train) saved in array format

    print('Starting data load and processing (normalize)...')
    input_size = 56 # 56, 128 or 200
    X_train = ed.load_Xdata('data/array_data/X_train_{}.npz'.format(input_size))

    # extract y labels sorted
    Ids = pd.read_csv('data/train.csv')
    ImageToLabelDict = dict(zip(Ids['Image'], Ids['Id']))
    sorted_image = list()
    train_images = ed.data_train_extraction('data/train')
    test_images = ed.data_train_extraction('data/test')
    for img in train_images:
        sorted_image.append(ImageToLabelDict[img])
    df_train = pd.DataFrame({'Image': train_images, 'Id': sorted_image})
    # Delete images with labels 'new_whale'
    df_train = df_train[df_train['Id'] != 'new_whale']

    # Select Y labels to train
    y = df_train['Id']

    # Label encoder and one hot encoder. Necessary for Keras models
    y_label = ed.label_encoder.fit_transform(y)
    y_one_hot = ed.label_binarizer.fit_transform(y_label)

    # ---------------------------TRAIN------------------------------------------
    print('TRAIN')
    # Normalize images
    X_normalized = ed.normalize_images(X_train)
    preds_models = dict()
    for mdl in normalized_models:
        print('Predictions for {} model...'.format(mdl))
        preds = mdl.predict_proba(X_normalized)
        preds_models[mdl] = preds
    # --------------------------------------------------------------------------
    # Grey scale and normalize images
    del X_normalized
    gc.collect()
    X_normalized = ed.grey_normalize_images(X_train)
    X_normalized = X_normalized.reshape(X_normalized.shape[0], 1, 56, 56)
    for mdl in grey_normalized_models:
        print('Predictions for {} model...'.format(str(mdl)))
        preds = mdl.predict_proba(X_normalized)
        preds_models[mdl] = preds
    del X_train
    gc.collect()
    del X_normalized
    gc.collect()
    # --------------------------------------------------------------------------
    # Create preds DF
    df_preds_model_n = dict()
    n_classes = len(df_train['Id'].unique())
    print('Creating ensemble data frame')
    sum_preds = 0
    for i in range(n_classes):
        list_sum_preds = list()
        for j in range(len(preds)):
            for mdl in preds_models:
                sum_preds += preds_models[mdl][j][i]
            list_sum_preds.append(sum_preds)
            sum_preds = 0
        df_preds_model_n[str(i)] = list_sum_preds

    df_preds_model_n['Id'] = y
    df_preds_train = pd.DataFrame(df_preds_model_n)
    df_preds_train.to_csv('data/train_ensemble_1.csv', index=False)

    del df_preds_train
    gc.collect()

    # ---------------------------TEST-------------------------------------------
    print('')
    print('TEST')
    X_test = ed.load_Xdata('data/array_data/X_test_{}.npz'.format(input_size))
    # Normalize images
    X_normalized = ed.normalize_images(X_test)
    preds_models = dict()
    for mdl in normalized_models:
        print('Predictions for {} model...'.format(mdl))
        preds = mdl.predict_proba(X_normalized)
        preds_models[mdl] = preds
    # --------------------------------------------------------------------------
    # Grey scale and normalize images
    del X_normalized
    gc.collect()
    X_normalized = ed.grey_normalize_images(X_test)
    X_normalized = X_normalized.reshape(X_normalized.shape[0], 1, 56, 56)
    for mdl in grey_normalized_models:
        print('Predictions for {} model...'.format(str(mdl)))
        preds = mdl.predict_proba(X_normalized)
        preds_models[mdl] = preds
    del X_test
    gc.collect()
    del X_normalized
    gc.collect()
    # --------------------------------------------------------------------------
    # Create preds DF
    df_preds_model_n = dict()
    n_classes = len(df_train['Id'].unique())
    print('Creating ensemble data frame')
    sum_preds = 0
    for i in range(n_classes):
        list_sum_preds = list()
        for j in range(len(preds)):
            for mdl in preds_models:
                sum_preds += preds_models[mdl][j][i]
            list_sum_preds.append(sum_preds)
            sum_preds = 0
        df_preds_model_n[str(i)] = list_sum_preds

    df_preds_train = pd.DataFrame(df_preds_model_n)
    df_preds_train.to_csv('data/test_ensemble_1.csv', index=False)

    print('Ensemble_1 dataset generated in: {} min'.format(round((time.time() - t0)/60, 1)))
    # mean elapsed time to execute this script (1 thread): 24 min