# Author: Shailja Thakur
# MIRROR: Model Interpretability using Reconstruction Error of sub-sampled input 

import numpy as np 
import pandas as pd 
import os 
from utils import read_dataset, reshape, label_encoding, calculate_metrics, smooth, norm, create_directory 
from scipy.signal import savgol_filter
import tensorflow.keras as keras
# import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt 
import glob
from tensorflow.keras.models import model_from_json
from autograd import grad
import autograd.numpy as np

dir = '/rhome/s7thakur/'
# dataset_name = "ItalyPowerDemand"
N = 1000
w_size = 20
momentum = 0.005

def read_model(load_path, model):
    
    # load json and create model
    json_file = open(os.path.join(load_path, model+'.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(load_path, model+".h5"))
    print("Loaded model from disk")
    
    return loaded_model


def load_model(model_dir, dataset_name, model_name = 'ResNet1D', load=True):

    """ Function returning keras model instance

    Model can be
    - Trained here
    - Loaded with load_model
    - Loaded from keras.applications
    """

    if load == 0:
        # Train the model 
        pass
    if load == 1:
        # load the saved model instance
        if os.path.exists(os.path.join(model_dir, model_name, dataset_name, 'best_model.hdf5')):
            model_path = os.path.join(model_dir, model_name, dataset_name, 'best_model.hdf5')
            model = keras.models.load_model(model_path)
        else:
            
            model_path = os.path.join(model_dir, model_name, dataset_name, 'last_model.hdf5')
            model = keras.models.load_model(model_path)
            
        print(model_path)
        if os.path.exists(os.path.join(model_dir, 'vae', dataset_name, 'best_vae.h5')):
            autoencoder_dir = os.path.join(model_dir, 'vae', dataset_name, 'best_vae.h5')
            autoencoder = read_model(os.path.join(model_dir,'vae', dataset_name), 'best_vae')
#             encoder = read_model(os.path.join(model_dir,'vae', dataset_name), 'best_encoder')
#             decoder = read_model(os.path.join(model_dir,'vae', dataset_name), 'best_decoder')
#             autoencoder = keras.models.load_model(autoencoder_dir)
        else:
            autoencoder_dir = os.path.join(model_dir, 'vae', dataset_name, 'last_vae.h5')
            autoencoder = read_model(os.path.join(model_dir,'vae', dataset_name), 'last_vae')
#             encoder = read_model(os.path.join(model_dir,'vae', dataset_name), 'best_decoder')
#             decoder = read_model(os.path.join(model_dir,'vae', dataset_name), 'best_decoder')
#             autoencoder = keras.models.load_model(autoencoder_dir)
            
        return model, autoencoder

    if load == 2:
        # Load the pre-trained model instance from keras.applications 
        pass


def load_time_series():

    # Load and preprocess the input time-series

    x_test, y_test = read_dataset(dir, dataset_name, 'TEST.tsv')
    x_test = reshape(x_test)

    y_test = label_encoding(y_test)

    return x_test, y_test

# 
def gen_mask(N, s):
    grids = np.empty((N, s))
    samples = np.random.normal(0, 1, size=(N, s))
    samples = samples.astype('float32')

    for i in range(0, N):
        grids[i] = smooth(samples[i], w_size)[0:s]

    return grids

def calculate_mse(x, y, recon_pred, class_idx, param):
 
    x = x.reshape(x.shape[0], x.shape[1])
    y = y.reshape(y.shape[0], y.shape[1])
    mse = np.square(np.abs(y-x))*param 
    return mse 

def mirror_batch(x, classes, model, autoencoder):

    # Initialize a variable of the same dimension as the input to store the 
    # saliency of the batch of input time-series 
    sal = np.empty((x.shape[0], x.shape[1], 1))

    # generate saliency map for each of the individual input time-series
    for i, test in enumerate(x):

        samples = gen_mask(N, x.shape[1])
        samples = reshape(samples)
        # Normalize\
        test = norm(test)
        # print('Data after normalization',test.shape)
        # Input sub-samples
        masked_test = samples*test + samples 

        masked_test = (masked_test - np.mean(masked_test))/np.std(masked_test)
        true_pred = model.predict(test.reshape(1, x.shape[1],1))
        class_idx = np.argmax(true_pred)
        # print(masked_test.shape)

        # Reconstructing input-samples
#         print(masked_test[:,:,0].shape)
#         print(autoencoder.summary())
        recon = autoencoder.predict(masked_test[:,:,0])
        
        recon_pred = model.predict(recon.reshape(N, x.shape[1], 1))

        # Reconstruction error on the input sub-samples
        mse = calculate_mse(recon, samples, recon_pred, class_idx, momentum)
        sal[i] = np.sum(mse*masked_test.reshape(masked_test.shape[0],
            masked_test.shape[1]), axis=0).reshape(sal.shape[1], 1)


    return sal 



def visualize_saliency(x_test, y_test, classes, sal, save_path, window_len=51):

    window_len = x_test.shape[1] - 10
    if window_len%2 == 0:
        window_len = window_len+1

    y_test = np.expand_dims(y_test, axis=1)
    print(x_test.shape, y_test.shape)
    sal = np.abs(sal)
    Y = np.argmax(y_test, axis=1)
    # print(x.size)

    for c in classes:

        plt.figure(figsize=(10,5))        
        c_x = x_test[np.where(Y == c)]
        c_sal = sal[np.where(Y==c)]

        for sa, test_input in zip(c_sal, c_x):

            sa = savgol_filter(sa[:,0], window_len, 1) # window size 51, polynomial order 3
            max_length = 2000
            minimum = np.min(sa)

            sa = sa - minimum
            sa = sa / np.max(sa)
            sa= sa * 100

            ts = test_input.reshape(1, x_test.shape[1],1)
            x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
            f = interp1d(range(ts.shape[1]), ts[0, :, 0])
            y = f(x)
            f = interp1d(range(ts.shape[1]), sa)
            cas = f(x).astype(int)
            plt.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=20, vmin=0, vmax=100, linewidths=0.0)
            plt.title('class:{}'.format(c), fontsize=20)

        print('result plot save file:', os.path.splitext(save_path)[0] + '_class_' + str(c) + '.png')
        plt.savefig(os.path.splitext(save_path)[0] + '_class_' + str(c) + '.png', 
            transparent=True, bbox_inches='tight',pad_inches=0)    
        # plt.show()


def compute_saliency(x_test, y_test, classes, dataset_name, save_fig_path, save_sal_path, model, autoencoder, visualize=True, save=True):

    predictions = model.predict(reshape(x_test))

    class_idx = np.argmax(predictions[0])
    class_name = classes[class_idx]
    # print("Explanation for '{}'".format(class_name))

    sal = mirror_batch(x_test, classes, model, autoencoder)

    if save:

        np.save(save_sal_path, sal)

    if visualize:
        visualize_saliency(x_test, y_test, classes, sal, save_fig_path)


if __name__ == "__main__":

    data_path = os.path.join(dir, 'UCRArchive', 'UCRArchive_2018')
    save_dir = os.path.join(dir, 'MIRROR', 'logs')
    model_path = os.path.join(dir, 'time-series-models')
#     model_path = os.path.join(dir, 'ResNet1D')
#     dataset_names = pd.read_csv(os.path.join(dir,'ResNet1D', 'DataSummary.csv'))['Name']
    dataset_names = ['PhalangesOutlinesCorrect', 'CinCECGTorso','ItalyPowerDemand', 'Trace', 'GunPoint', 'GunPointAgeSpan', 
                            'Strawberry', 'ECGFiveDays', 'TwoLeadECG', 'Chinatown', 'DistalPhalanxOutlineCorrect']

#     dataset_names = glob.glob(os.path.join('/rhome/s7thakur/time-series-models/vae/*'))
#     dataset_names = [f.split('/')[5] for f in dataset_names]

    models = ['cnn', 'inception', 'resnet']
#     models = ['resnet']
#     dataset_names = ['PhalangesOutlinesCorrect', 'CinCECGTorso','ItalyPowerDemand', 'Trace', 'GunPoint', 'GunPointAgeSpan', 
#                         'Strawberry', 'ECGFiveDays', 'TwoLeadECG', 'Chinatown', 'DistalPhalanxOutlineCorrect']

    for dataset_name in dataset_names:

        print('dataset:', dataset_name)
        # Read time-series dataset
        x_test,y_test = read_dataset(data_path, dataset_name)
        x_train,y_train = read_dataset(data_path, dataset_name, 'TRAIN.tsv')

        if x_train.shape[0] < x_test.shape[0]:
            x = x_train[0:50]
            y = y_train[0:50]
        else:
            x = x_test[0:50]
            y = y_test[0:50]

        # Load black-box model, whose output we want to interpret, and 
        # Load the autoencoder for calculating reconstruction error on the sub-sampled inputs

        for model_name in models:
            print('model:', model_name)
            model, autoencoder = load_model(model_path, dataset_name, model_name)
            classes = np.unique(np.concatenate((y_train, y_test), axis=0))
            # print('Unique classes:', classes)
            save_fig_path = os.path.join(save_dir, 'vae', 'figures',dataset_name + '-' + model_name +'.npy')
            save_sal_path = os.path.join(save_dir, 'vae', 'saliency',dataset_name + '-' + model_name +'.npy')
            
            sal = compute_saliency(x, y, classes, dataset_name, save_fig_path, save_sal_path, model, autoencoder)




