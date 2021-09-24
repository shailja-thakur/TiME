import numpy as np
import pandas as pd
import os
import operator
import re
import utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt 
import seaborn as sns 
from utils.constants import ARCHIVE_NAMES  as ARCHIVE_NAMES
from utils.constants import CLASSIFIERS
from utils.constants import ITERATIONS
from utils.constants import DATASET_EVALUATION
from scipy import signal
from sklearn import preprocessing
from collections import Counter
import torch

# def read_dataset(archive_dir, dataset_name, data_type='TEST.tsv'):
    
#     # Read the time-series dataset as a dataframe
#     df = pd.read_csv(os.path.join(archive_dir, dataset_name) + '/' + dataset_name + '_' + data_type,
#                         sep='\t', header=None)

#     # Fetch the column with labels from the loaded dataframe
#     y = df.values[:,0]
#     # Drop the column with labels
#     x = df.drop(columns=[0])    
#     # Convert the data to numpy array
#     x = x.values
    
#     # znorm
#     std_ = x.std(axis=1, keepdims=True)
#     std_[std_ == 0] = 1.0
#     x = (x - x.mean(axis=1, keepdims=True)) / std_
    
#     return x, y

def label2index(i):
    # m = {'N':0, 'S':1, 'V':2, 'F':3, 'Q':4} # uncomment for 5 classes
    m = {'N':0, 'S':0, 'V':1, 'F':0, 'Q':0} # uncomment for 2 classes
    return m[i]

def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')

    if archive_name == 'mts_archive':
        file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
        x_train = np.load(file_name + 'x_train.npy')
        y_train = np.load(file_name + 'y_train.npy')
        x_test = np.load(file_name + 'x_test.npy')
        y_test = np.load(file_name + 'y_test.npy')

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    elif archive_name == 'UCRArchive/UCRArchive_2018':
        print(archive_name)
        # root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name 
        root_dir_dataset = cur_root_dir + '/'+ archive_name + '/' + dataset_name 

        df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

        df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]

        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    else:
        file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
        x_train, y_train = readucr(file_name + '_TRAIN')
        x_test, y_test = readucr(file_name + '_TEST')
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    return datasets_dict

def read_dataset_mitdb(root_dir, archive_name, dataset_name, MyDataset, DataLoader, batch_size = 1):

    # read data
    print('start')

    data = np.load(os.path.join(root_dir, archive_name, 'data','mitdb_data.npy'))
    label_str = np.load(os.path.join(root_dir, archive_name, 'data','mitdb_group.npy'))
    label = np.array([label2index(i) for i in label_str])
    
    
    # make data
    train_ind = np.load(os.path.join(root_dir, archive_name, 'data','mitdb_train_ind.npy'))
    test_ind = np.load(os.path.join(root_dir, archive_name, 'data','mitdb_test_ind.npy'))

    
    # data = preprocessing.scale(data, axis=1)
    # X_train = data[train_ind]
    X_test = data[test_ind]
    # Y_train = label[train_ind]
    Y_test = label[test_ind]


    print(X_test.shape,Y_test.shape)

    # znorm
    # std_ = X_train.std(axis=1, keepdims=True)
    # std_[std_ == 0] = 1.0
    # X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / std_

    std_ = X_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    X_test = (X_test - X_test.mean(axis=1, keepdims=True)) / std_

    # print(X_train.shape, Counter(Y_train))
    # print(X_test.shape, Counter(Y_test))
    # ros = RandomOverSampler(random_state=0)
    # X_train, Y_train = ros.fit_resample(X_train, Y_train)
    # print(X_train.shape, Counter(Y_train))
    # print(np.max(X_train), np.min(X_train))
    
    
    
    # prepare loader
    # shuffle_idx = np.random.permutation(list(range(X_train.shape[0])))
    # X_train = X_train[shuffle_idx]
    # Y_train = Y_train[shuffle_idx]
    # X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)
    # X_test = norm(X_test)

    # dataset = MyDataset(X_train, Y_train)
    dataset_test = MyDataset(X_test, Y_test)
    # dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False, shuffle=False)

    return dataloader_test, Y_test, X_test.shape, np.unique(label) 

def read_all_datasets(root_dir, archive_name, split_val=False):
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []

    if archive_name == 'mts_archive':

        for dataset_name in MTS_DATASET_NAMES:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset + 'x_train.npy')
            y_train = np.load(root_dir_dataset + 'y_train.npy')
            x_test = np.load(root_dir_dataset + 'x_test.npy')
            y_test = np.load(root_dir_dataset + 'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
    elif archive_name == 'UCRArchive':
        for dataset_name in DATASET_EVALUATION:
            root_dir_dataset = cur_root_dir + '/' + archive_name + '/'+ 'UCRArchive_2018' + '/' + dataset_name 
            # root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name 

            df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

            df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    else:
        for dataset_name in DATASET_NAMES:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN')
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict

def znorm(x_train, ax):

    # znorm
    std_ = x_train.std(axis=ax, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=ax, keepdims=True)) /std_

    return x_train

def torch_norm(x):


    x = (x - torch.mean(x))/torch.std(x)
    return x.type(torch.cuda.FloatTensor).cuda() 

def norm(x):

    x = (x - np.mean(x))/np.std(x)

    return x
    
def label_encoding(y):
    
    # transform the labels from integers to one hot vectors
    enc = OneHotEncoder(categories='auto')
    enc.fit(y.reshape(-1, 1))
    y = enc.transform(y.reshape(-1, 1)).toarray()

    return y

def reshape(x):

    if len(x.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x = x.reshape((x.shape[0], x.shape[1], 1))

        return x

    else:
        print('Input is not univariate')
        return 

def save_saliency(file_name, saliency):

    if  os.path.exists(file_name):
        return
    else:
        np.save(file_name, saliency)


def calculate_metrics(y_true, y_pred, dataset_name):

    # Input: the predicted (y_pred) and true (y_true) labels of the input time-series
    # Output: Return a dataframe with precision, recall, accuracy and the dataset name

    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['dataset','precision', 'accuracy', 'recall'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['dataset'] = dataset_name
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def save_logs(archive_name, dataset_name, classifier_name, file_name, sal, preds, test_duration,  
                window_name=utils.constants.window_name,lobe=utils.constants.max_lobe,
                window_size=utils.constants.window_size, epoch=utils.constants.epoch, 
                batch=utils.constants.batch):
    
    sal_log = dict()

    keys = ['archive_name','dataset_name','classifier_name','rise_fall_edges','window_name','window_val','lr', 
            'epochs','batches','importance_scores','preds_scores','latency',
            'notes']

    # notes: 'prediction on smallest chunk of test/train chunks'

    values = [archive_name, dataset_name, classifier_name, lobe, window_name, 
                window_size,utils.constants.lr, epoch, batch, sal, preds, 
                test_duration, utils.constants.notes]

    # print(len(keys), len(values))

    sal_log = dict(zip(keys, values))
    np.save(file_name, sal_log)

    print('Saving salincy done')



def generate_results_csv(output_file_name, root_dir, archive_name):

    

    if archive_name == 'UCRArchive':

        res = pd.DataFrame(data=np.zeros((0, 7), dtype=np.float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'AUP', 'AUR', 'AUC (Insertion)', 'AUC (Deletion)','Coherency','Complexity','Duration', 'Train/Test', 'Avg Accuracy'])

        for classifier_name in CLASSIFIERS:
            for archive_name in ARCHIVE_NAMES:
                datasets_dict = read_all_datasets(root_dir, archive_name)
                for it in range(ITERATIONS):
                    curr_archive_name = archive_name
                    if it != 0:
                        curr_archive_name = curr_archive_name + '_itr_' + str(it)
                    for dataset_name in datasets_dict.keys():
                        output_dir = root_dir + '/results/' + classifier_name + '/' \
                                     + curr_archive_name + '/' + dataset_name + '/' + 'df_metrics.csv'
                        if not os.path.exists(output_dir):
                            continue
                        df_metrics = pd.read_csv(output_dir)
                        df_metrics['classifier_name'] = classifier_name
                        df_metrics['archive_name'] = archive_name
                        df_metrics['dataset_name'] = dataset_name
                        res = pd.concat((res, df_metrics), axis=0, sort=False)

        res.to_csv(root_dir + output_file_name, index=False)
        # aggreagte the accuracy for iterations on same dataset


        res = pd.DataFrame({
            'accuracy': res.groupby(
                ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
        }).reset_index()

        all_res = res.to_csv(os.path.join(root_dir, 'MIRROR','results','results_csv.csv'), mode='a')


    # if archive_name == 'hyperparams_eval':

    # if archive_name == 'comparison_approaches':




    # return res



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman','tukey']:
        raise ValueError


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')

    elif window=='tukey':
        w=signal.windows.tukey(window_len)
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def get_file_name(title, extension='pdf'):

	filename_keywords = re.sub('[^a-zA-Z0-9 \n\.]', '', title)
	filename_keywords = re.sub('\s+',' ',filename_keywords)
	filename_keywords = list(filename_keywords.split(' '))
	filename = '-'.join(filename_keywords)
	filename = '.'.join([filename, extension])

	return filename 


def plot_samples(test_input, samples, savepath, filename, classes, save=True, title=False, plot=False):

    fig=plt.figure()

    plt.plot(test_input,c='black')

    for sa in samples:
        
        plt.plot((norm(sa)))

    plt.legend(np.concatenate((['Input'], classes),axis=None))

    if(title):
        plt.title(filename)

    if(save):
    	
        filename=get_file_name(filename)
        fig.savefig(os.path.join(savepath,filename))

        plt.close(fig)
    if(plot):
        plt.show()

def plot_score_densities(preds, savepath, filename, save=True, title=True, plot=False):

    fig=plt.figure()
    preds = norm(preds)
    df = pd.DataFrame(preds, columns=list(map(str, np.arange(0,preds.shape[1]))))
    sns.displot(df,kind='ecdf')
    if save:
        filename = get_file_name(filename)
        fig.savefig(os.path.join(savepath,filename))
        plt.close(fig)
    if plot:
    
        plt.show()




