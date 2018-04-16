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
import time
import gc
from sklearn.externals import joblib
import lightgbm as lgb
from sklearn.metrics.pairwise import euclidean_distances
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

class mdl_ensemble(object):

    def __init__(self):
        if not os.path.exists('models/ensemble1_lightgbm'):
            os.makedirs('models/ensemble1_lightgbm')

        if not os.path.exists('submissions/ensemble1_lightgbm'):
            os.makedirs('submissions/ensemble1_lightgbm')

        if not os.path.exists('weights/ensemble1_lightgbm'):
            os.makedirs('weights/ensemble1_lightgbm')

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

    def submit(self, name, test_preds):
        with open('submissions/ensemble1_lightgbm/{}.csv'.format(name), 'w') as f:
            with warnings.catch_warnings():
                f.write("Image,Id\n")
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                n = 0
                for image in range(len(test_preds)):
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

    def new_submit(self,name, test_preds, thr=0.5):
        with open('submissions/ensemble1_lightgbm/{}.csv'.format(name), 'w') as f:
            with warnings.catch_warnings():
                f.write("Image,Id\n")
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                n = 0
                for image in range(len(test_preds)):
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

    def submit_euc_dist(self, name, X, y, test_images, test_preds):
        with open('submissions/ensemble1_lightgbm/{}.csv'.format(name), 'w') as f:
            with warnings.catch_warnings():
                f.write("Image,Id\n")
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                n = 0
                for image in test_preds:
                    # pred = test_preds.iloc[image, :]
                    # pred = test_preds[n]
                    euc_dist = euclidean_distances(X, image)
                    most_proba = list()
                    maxi_1 = max(image)
                    maxi_1_value = np.where(image == maxi_1)[0]
                    maxi_1_whale = self.label_encoder.inverse_transform(maxi_1_value)[0]
                    # most_proba.append(int(maxi_1_value[0]))
                    most_proba.append(maxi_1_whale)
                    # siguientes_2 = test_preds[np.where(test_preds != maxi_1)[0]]
                    maxi_1 = min(euc_dist)
                    maxi_1_value = np.where(euc_dist == maxi_1)[0]
                    most_proba.append(y[int(maxi_1_value[0])])
                    siguientes_2 = euc_dist[np.where(euc_dist != maxi_1)[0]]

                    maxi_2 = min(siguientes_2)
                    maxi_2_value = np.where(euc_dist == maxi_2)[0]
                    most_proba.append(y[int(maxi_2_value[0])])
                    siguientes_3 = siguientes_2[np.where(siguientes_2 != maxi_2)[0]]

                    maxi_3 = min(siguientes_3)
                    maxi_3_value = np.where(euc_dist == maxi_3)[0]
                    most_proba.append(y[int(maxi_3_value[0])])
                    # siguientes_4 = siguientes_3[np.where(siguientes_3 != maxi_3)[0]]

                    # maxi_4 = min(siguientes_4)
                    # maxi_4_value = np.where(euc_dist == maxi_4)[0]
                    # most_proba.append(int(maxi_4_value[0]))

                    most_proba = np.array(most_proba)
                    # most_proba = ['new_whale'] + most_proba.tolist()
                    # label_arg = self.predict_4_better(test_preds[n - 1])
                    # preds = self.label_encoder.inverse_transform(most_proba)
                    preds = ['new_whale'] + most_proba.tolist()
                    predicted_tags = " ".join(preds)
                    f.write("%s,%s\n" % (test_images[n], predicted_tags))
                    n += 1
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

    # Label encoder y one hot. Necessary for keras models
    y_label = mdl.label_encoder.fit_transform(y)
    y_one_hot = mdl.label_binarizer.fit_transform(y_label)

    del train
    gc.collect()

    t0 = time.time()
    n_output = len(y.unique())

    # NAMES #
    model_name = 'model_ensemble1_1'
    submit_name = 'submission_ensemble1'
    #########
    print('Generating model {}...'.format(model_name))

    # fit_dict_xgbc = {
    #     # "eval_set": [(X_train, y_train)],
    #                  "early_stopping_rounds": 5,
    #                  "verbose": True,
    #                  "eval_metric": "mlogloss",
    #                  }
    #
    #
    # parameters_xgbc = {"learning_rate": [0.05],
    #                    "max_depth": [10, 20],
    #                    "n_estimators": [500, 600],
    #                    }
    #
    #
    # xgboost_estimator = XGBClassifier(nthread=4,
    #                               seed=42,
    #                               subsample=0.8,
    #                               colsample_bytree=0.6,
    #                               colsample_bylevel=0.7,
    #                               )
    #
    # model = GridSearchCV(estimator=xgboost_estimator,
    #                              param_grid=parameters_xgbc,
    #                              n_jobs=4,
    #                              cv=5,
    #                              fit_params=fit_dict_xgbc,
    #                              verbose=10,
    #                              )
    # model = XGBClassifier

    train_data = lgb.Dataset(X, label=y_label)
    del X
    gc.collect()

    #
    # Train the model
    #

    parameters = {
        'application': 'multiclass',
        'objective': 'multiclass',
        'metric': 'auc',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'feature_fraction': 1,
        # 'bagging_fraction': 0.5,
        # 'bagging_freq': 20,
        'learning_rate': 0.1,
        'verbose': 100,
        'num_class': n_output,
        'max_depth': 5
    }

    model = lgb.train(parameters,
                           train_data,
                           num_boost_round=100,
                           # early_stopping_rounds=10
                      )

    # model.fit(X, y_label,
    #                   # evals=[(dtest, 'test')],
    #                   # evals_result=gpu_res
    #                   )

    joblib.dump(model, 'models/ensemble1_lightgbm/{}.pkl'.format(model_name))

    # from keras.models import load_model
    # model = load_model('models/normalize/model_whitout_new_whale_56_6.h5')
    # model.load_weights('weights/normalize/model_whitout_new_whale_56_6.h5')
    print('Training time: {} min'.format(round((time.time() - t0) / 60, 2)))

    del X
    gc.collect()
    print('Predictions...')
    X_test = pd.read_csv('data/test_ensemble_1.csv')
    test_preds = model.predict_proba(X_test)
    del X_test
    gc.collect()

    t0 = time.time()
    # mdl.submit_euc_dist(submit_name+'_euc_dist', X, y, test_images, test_preds)
    mdl.new_submit(submit_name+'_thr099', test_preds, thr=0.99)
    mdl.submit(submit_name, test_preds)
    print('Submit prediction in: {} secs'.format(round(time.time() - t0, 1)))