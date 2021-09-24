# Author: Shailja Thakur
# MIRROR: Model Interpretability using Reconstruction Error of sub-sampled input 
import sys
import numpy as np 
import pandas as pd 
import os 
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

__file__ = '/home/s7thakur/ecresearch-shared/PhysioNet/resnet1d/resnet1d'
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from resnet1d.resnet1d import MyDataset, ResNet1D

from utils.utils import reshape, label_encoding, calculate_metrics, smooth, norm
import sklearn
import tensorflow.keras as keras
# import tensorflow as tf
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt 
from tensorflow.keras.models import model_from_json
import platform 
import utils
import time
import explain
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.utils import read_all_datasets, create_directory, read_dataset, save_logs, read_dataset_mitdb
import os.path as path
import matplotlib as mpl
import utils.constants
import torch
import torch.nn as nn
import torch.nn.functional as F



os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

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


def load_model(model_directory, load=True):

    """ Function returning keras model instance

    Model can be
    - Trained here
    - Loaded with load_model
    - Loaded from keras.applications
    """


    model_best_path = os.path.join(model_directory, 'best_model.hdf5')
    model_last_path = os.path.join(model_directory, 'last_model.hdf5')
    print(model_best_path)
    print(model_last_path)
    if os.path.exists(model_best_path):
        print('best model ')
        model = keras.models.load_model(model_best_path)
        return model
    elif os.path.exists(model_last_path):
        
        model = keras.models.load_model(model_last_path)
        return model
    else:
        print("No model exist for dataset={} and classifier={}".format(dataset_name, classifier_name))
        return None
  



def visualize_saliency(x_test, y_test, classes, sal, save_path, window_len=5):

#     window_len = x_test.shape[1] - 10
#     if window_len%2 == 0:
#         window_len = window_len+1

    y_test = np.expand_dims(y_test, axis=1)
    print(x_test.shape, y_test.shape)
    sal = np.abs(sal)
    Y = np.argmax(y_test, axis=1)
    # print(x.size)

    for c in classes:

        fig=plt.figure(figsize=(10,5))        
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
        plt.close(fig)
        # plt.show()


def save_saliency_plot(X, Y, classes,sal, save_path, save=True):

    print('Y shape',Y.shape)
    Y_ids = np.argmax(Y,axis=1)
    print('Y=',Y_ids)
    # Y = Y[:,Y_ids]
    classes = classes - 1 

    fig, axes = plt.subplots( ncols=len(classes), figsize=(12, 4))
    # fig.subplots_adjust(hspace=0.5)

    for c,ax in zip(classes,axes):
        
    #     plt.figure(figsize=(10,5))        
        c_x = X[np.where(Y == c)]
        c_sal = sal[np.where(Y==c)]
        print(c_x.shape, c_sal.shape)
        for sa, test_input in zip(c_sal, c_x):
            print('sa shape',sa.shape, c)
            sa = savgol_filter(sa[:,c], 5, 1) # window size 51, polynomial order 3
            max_length = 6000
            minimum = np.mean(sa)
            
            sa = sa - minimum
            sa = sa / np.max(sa)
            sa= sa * 100
            
            ts = test_input.reshape(1, test.shape[1],1)
            x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
            f = interp1d(range(ts.shape[1]), ts[0, :, 0])
            y = f(x)
            
            f = interp1d(range(ts.shape[1]), sa)
            cas = f(x).astype(int)
             
            ax.scatter(x,y, c=cas, cmap='jet', marker='.', s=20, vmin=0, vmax=100, linewidths=0.0)
        ax.set_title('class:{}'.format(c), size=25) 
        ax.set_ylabel('Saliency Score', size=25) #setting the ylabel and font size
        ax.set_xlabel('Index', size=25)
        ax.xaxis.set_tick_params(labelsize=25) #setting the font size of the x axis
        ax.yaxis.set_tick_params(labelsize=25) #setting the font size of the y axis
    
    # plt.tight_layout()

    if save:
        fig.savefig(os.path.join(save_path,'ImportanceScore_plot.pdf'), transparent=True, bbox_inches='tight',pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def save_saliency_discirminations_plot(X, Y, classes, sal, save_path, save=True):

    Y = np.argmax(Y,axis=1)


    fig, axes = plt.subplots(nrows=len(classes), ncols=len(classes), figsize=(20, 15))

    for c_i in classes:
        
        c_x = X[np.where(Y == c_i)]
        c_sal = sal[np.where(Y==c_i)]
            
        for c_j in classes:
            for sa, test_input in zip(c_sal, c_x):
                sa = savgol_filter(sa[:,c_j], 11, 1) # window size 51, polynomial order 3
                max_length = 6000
                minimum = np.mean(sa)

                sa = sa - minimum
                sa = sa / np.max(sa)
                sa= sa * 100

                ts = test_input.reshape(1, test.shape[1],1)
                x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                y = f(x)
                f = interp1d(range(ts.shape[1]), sa)
                cas = f(x).astype(int)
                axes[c_i,c_j].scatter(x,y, c=cas, cmap='jet', marker='.', s=20, vmin=0, vmax=100, linewidths=0.0)
            axes[c_i,c_j].set_title('True Class:{}, Sal Class:{}'.format(c_i,c_j), size=25) 
            axes[c_i,c_j].set_ylabel('Saliency Score', size=15) #setting the ylabel and font size
            axes[c_i,c_j].set_xlabel('Index', size=15)
            axes[c_i,c_j].xaxis.set_tick_params(labelsize=15) #setting the font size of the x axis
            axes[c_i,c_j].yaxis.set_tick_params(labelsize=15) #setting the font size of the y axis
        
    if save:
        fig.savefig(os.path.join(save_path, 'class-discriminative-ImportanceScore.pdf'), transparent=True, bbox_inches='tight',pad_inches=0)
        plt.close(fig)
    else:
        plt.show()


def compute_attributions_mitbh(loaders, labels, input_shape, window_name=utils.constants.window_name, 
                            window_size=utils.constants.window_size, lobe=utils.constants.max_lobe,
                            batch=utils.constants.batch,epochs=utils.constants.epoch, visualize=False, save=True):

    print('save log',os.path.join(output_directory,'importance_score.npy'))


    # make model
    device_str = "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # if isinstance(labels, np.ndarray):
    #     nb_classes = np.unique(labels)
    # else:
    #     nb_classes = np.unique(labels.cpu().numpy())

    nb_classes = classes
    model = ResNet1D(
        in_channels=1, 
        base_filters=128, 
        kernel_size=16, 
        stride=2, 
        groups=8, 
        n_block=8, 
        n_classes=len(classes), 
        downsample_gap=2, 
        increasefilter_gap=4, 
        use_do=False)

    # summary(model, torch.zeros(1, 1, 360))
    model.to(device)

    # define optimzer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # define checkpoint saved path
    ckp_path = os.path.join(model_directory,"current_checkpoint_two_class.pt")

    # load the saved checkpoint
    model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)

    # print("model = ", model)
    print("optimizer = ", optimizer)
    print("start_epoch = ", start_epoch)
    print("valid_loss_min = ", valid_loss_min)
    print("valid_loss_min = {:.6f}".format(valid_loss_min))

    model.eval()

    test_acc = 0.0
    # for samples, labels in loaders:
    #     with torch.no_grad():
    #         samples, labels = samples.cuda(), labels.cuda()
    #         # print(samples.shape, labels.shape)
    #         output = model(samples)
    #         # calculate accuracy
    #         pred = torch.argmax(output, dim=1)
    #         correct = pred.eq(labels)
    #         test_acc += torch.mean(correct.float())
    #         # print('output {}'.format(output))
    #         # print('correct {}'.format(correct))
    #         # print('predict {}'.format(pred))

    #         # print('test Accuracy {}'.format(test_acc))

    # print('Accuracy of the network on {} test images: {}%'.format(len(labels), round(test_acc.item()*100.0/len(loaders), 2)))


    # model = load_model(model_directory)
    
    if model ==None: return
    

   
    input_shape = input_shape[1:]
    
    
    N=len(loaders)


    explainer = explain.Explain(loaders, input_shape, N, output_directory, model, nb_classes)

    print(input_shape[1], utils.constants.window_size)

    if int(input_shape[1]) >= utils.constants.window_size:

        sal, preds, test_duration = explainer.get_imp_weight_vector_physionet()


        if save:
            print('SAVING ...')

            save_logs(archive_name, dataset_name, classifier_name, 
                os.path.join(output_directory,'importance_score.npy'), 
                sal, preds, test_duration, window_name,
                lobe, window_size, epochs, batch)

            print('SAVING DONE!!')
            # save_saliency_plot(x, y, nb_classes, sal, output_directory)
            # save_saliency_discirminations_plot(x, y, nb_classes, sal, output_directory)

        if visualize:
            visualize_saliency(x_test, y_test, nb_classes, sal, output_directory)



def compute_attributions(window_name=utils.constants.window_name, 
                            window_size=utils.constants.window_size, lobe=utils.constants.max_lobe,
                            batch=utils.constants.batch,epochs=utils.constants.epoch,
                            visualize=False, save=True):

    if save:
        print('save log',os.path.join(output_directory,'importance_score.npy'))
    else:
        print('save:', save)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    
    model = load_model(model_directory)
    print('Loaded Model')
    if model ==None: return
    nb_classes = (np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    if x_train.shape[0] < x_test.shape[0]:
        x = x_train
        y = y_train
    else:
        x = x_test
        y = y_test
    
   
    input_shape = x.shape[1:]
    
    y_preds = model.predict(x)
    
    
    class_idx = np.argmax(y_preds[0])
    
    #class_name = nb_classes[class_idx]

    N=x.shape[0]

    if sys.argv[1] == 'hyperparams_eval':

        explainer = explain.Explain(x, input_shape, N, output_directory, model, 
            nb_classes, window_name, window_size, lobe,batch, epochs)
    else:
        explainer = explain.Explain(x, input_shape, N, output_directory, model, nb_classes)

    print(input_shape[0], window_size)

    if int(input_shape[0]) >= window_size:

        sal, preds, test_duration = explainer.get_imp_weight_vector()


        if save:
            print('SAVING ........')
            save_logs(archive_name, dataset_name, classifier_name, 
                os.path.join(output_directory,'importance_score.npy'), 
                sal, preds, test_duration,window_name,
                lobe, window_size, epochs, batch)
            print('SAVING DONE!!')
            # save_saliency_plot(x, y, nb_classes, sal, output_directory)
            # save_saliency_discirminations_plot(x, y, nb_classes, sal, output_directory)

        if visualize:
            visualize_saliency(x_test, y_test, nb_classes, sal, output_directory)
        

def get_root_dir():


    if platform.node() == "compute3.esg.uwaterloo.ca":
        if os.listdir('/home/s7thakur/ecresearch-shared'):
            filepath = '/home/s7thakur/ecresearch-shared'

        else: 
            print('ecresearch-shared is not mounted')

            # mounting ecresearch and re-trying


    
    elif platform.node() == "compute1.esg.uwaterloo.ca":

        if os.listdir('/rhome/s7thakur/ecresearch-shared'):
            filepath = '/rhome/s7thakur/ecresearch-s7thakur'
            print(filepath)
        else: 
            print('ecresearch-shared is not mounted')

    elif platform.node() == 'phi2@esg.uwaterloo.ca':

        if os.listdir('/home/s7thakur/ecresearch-shared'):

            filepath='/home/s7thakur/ecresearch-shared'
    else:
        filepath='/home/s7thakur/ecresearch-shared/'
        print('enter the path to data and scripts correctly before running explainability algorithm')

    return filepath


def get_hyperparameters(param, val):

    param_dict=dict()

    param_dict['window_name'] = val if  param == 'WINDOW_NAMES' else utils.constants.window_name

    param_dict['window_size'] = val if  param == 'WINDOW_SIZES' else utils.constants.window_size
    
    param_dict['max_lobe'] = val if  param == 'MAX_LOBES' else utils.constants.max_lobe

    param_dict['batch'] = val if  param == 'BATCHES' else utils.constants.batch

    param_dict['epoch'] = val if  param == 'EPOCHS' else utils.constants.epoch


    return param_dict

# sal = np.ones((test.shape[0], test.shape[1],n_classes))      
# masked_preds_inputs = np.empty((test.shape[0],N,n_classes))

# for model_name in models:
#     print('model:', model_name)
#     model, autoencoder = load_model(model_path, dataset_name, model_name)
#     classes = np.unique(np.concatenate((y_train, y_test), axis=0))
#     # print('Unique classes:', classes)
#     save_fig_path = os.path.join(save_dir, 'vae', 'figures',dataset_name + '-' + model_name +'.npy')
#     save_sal_path = os.path.join(save_dir, 'vae', 'saliency',dataset_name + '-' + model_name +'.npy')
    
#     sal = compute_saliency(x, y, classes, dataset_name, save_fig_path, save_sal_path, model)



############################################### main

# change this directory for your machine

root_dir = get_root_dir()
print(root_dir)

tex_style_file = path.join(mpl.__path__[0], 'mpl-data', 'stylelib','tex.mplstyle')

if not os.path.exists(tex_style_file):

    tex = '''
        text.usetex: True
        font.family: serif
        axes.labelsize: 10
        font.size: 10
        legend.fontsize: 8
        xtick.labelsize: 8
        ytick.labelsize: 8
    '''

    f = open(tex_style_file, 'w')
    f.write(tex)
    f.close()


if sys.argv[1] == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)

            tmp_output_directory = os.path.join(root_dir, 'MIRROR','results' , 'ResNet1D',classifier_name, archive_name)
            tmp_model_directory = os.path.join('/home/s7thakur/ecresearch-shared/compute1/','ResNet1D', classifier_name)
            # tmp_model_directory = os.path.join(root_dir, 'time-series-models', classifier_name)

            print('temp tmp_output_directory', tmp_output_directory)

            for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                print('\t\t\tdataset_name: ', dataset_name)

                output_directory = os.path.join(tmp_output_directory , dataset_name) 
                model_directory=os.path.join(tmp_model_directory, dataset_name)

                print('output_directory',output_directory, model_directory)

                if os.path.exists(output_directory+'/DONE'): continue
                 
                create_directory(output_directory)


                compute_attributions()

                print('\t\t\t\tDONE')

                # the creation of this directory means
                create_directory(output_directory + '/DONE')

elif sys.argv[1] == 'performance_eval':


    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)


            # tmp_output_directory = os.path.join(root_dir, 'MIRROR','results' , 'ResNet1D',classifier_name, archive_name)
            tmp_model_directory = os.path.join(root_dir,'compute1','time-series-models', classifier_name)

            # print('temp tmp_output_directory', tmp_output_directory)

            for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                print('\t\t\tdataset_name: ', dataset_name)

                # output_directory = os.path.join(tmp_output_directory , dataset_name) 
                output_directory = None
                model_directory=os.path.join(tmp_model_directory, dataset_name)

                # print('output_directory',output_directory, model_directory)

                # if os.path.exists(output_directory+'/DONE'): continue
                 
                # create_directory(output_directory)


                compute_attributions(save=False)

                print('\t\t\t\tDONE')

                # the creation of this directory means
                #create_directory(output_directory + '/DONE')

elif sys.argv[1] == 'PhysioNet':

    # this is the code used to launch an experiment on a dataset
    # physioNet
    archive_name = sys.argv[1]
    # mitdb
    dataset_name = sys.argv[2]
    # resnet
    classifier_name = sys.argv[3]
    # itr = sys.argv[4]

    output_directory = os.path.join(root_dir ,'MIRROR', 'results' , classifier_name , archive_name, dataset_name) 
    model_directory=os.path.join(root_dir,'PhysioNet')
    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)
        loaders, labels, input_shape, classes = read_dataset_mitdb(root_dir, archive_name, dataset_name, MyDataset, DataLoader)

        compute_attributions_mitbh(loaders, labels, input_shape)

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')

elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1] == 'visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1] == 'viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1] == 'viz_cam':
    viz_cam(root_dir)
elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())

elif sys.argv[1] == 'hyperparams_eval':


    archive_name = 'UCRArchive'
    classifier_name = 'resnet'

    datasets_dict = read_all_datasets(root_dir, archive_name)

    for hyperparam in utils.constants.hyperparams.keys():

    # for window_name in utils.constants.WINDOW_NAMES:
    #     for window_size in utils.constants.WINDOW_SIZES:
    #         for lobe in utils.constants.MAX_LOBES:
    #             for batch in utils.constants.BATCHES:
    #                 for epoch in utils.constants.EPOCHS:  


        for val in utils.constants.hyperparams[hyperparam]:
        
            hyperparam_set = get_hyperparameters(hyperparam, val)
        
            hyper_params = '-'.join([str(hyperparam_set['window_name']), str(hyperparam_set['window_size']), 
                str(hyperparam_set['max_lobe']), str(hyperparam_set['batch']), str(hyperparam_set['epoch'])])

            print('hyperparameter set: ', hyper_params)

            tmp_output_directory = os.path.join(root_dir, 'MIRROR','results' , hyper_params, classifier_name, archive_name, 'UCRArchive_2018')
            tmp_model_directory = os.path.join('/home/s7thakur/ecresearch-shared/compute1/','ResNet1D', classifier_name)

            # print('temp tmp_output_directory', tmp_output_directory)
            # print('temp tmp_model_directory', tmp_model_directory)


            for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                print('\t\t\tdataset_name: ', dataset_name)

                output_directory = os.path.join(tmp_output_directory , dataset_name) 
                model_directory=os.path.join(tmp_model_directory, dataset_name)

                print('result path',output_directory)
                print('model path', model_directory)

                # if os.path.exists(output_directory+'/DONE'): continue
                 
                create_directory(output_directory)

                compute_attributions(window_name=hyperparam_set['window_name'], 
                            window_size=hyperparam_set['window_size'], lobe=hyperparam_set['max_lobe'],
                            batch=hyperparam_set['batch'],epochs=hyperparam_set['epoch'])

                print('\t\t\t\tDONE')

                # the creation of this directory means
                create_directory(output_directory + '/DONE')
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    # itr = sys.argv[4]

    output_directory = os.path.join(root_dir ,'MIRROR', 'results' , classifier_name , archive_name, dataset_name) 
    model_directory=os.path.join(root_dir,'time-series-models',classifier_name, dataset_name)
    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:

        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

        compute_attributions()

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')

