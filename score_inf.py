
"""
Train a SVM on normal frames and test the sensitivity of time
"""

import os, sys, collections, random, string
import numpy as np
import pdb
import pickle
from tempfile import TemporaryFile
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from svmutil import *
import sklearn.metrics as metrics
import classify_library
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import argparse

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

parser = argparse.ArgumentParser()

parser.add_argument('normal_train_data_dir', 
        default='./UCF101_Fishers/train', 
        type=str, 
        help='Dir of training fisher data')
parser.add_argument('shuffled_train_data_dir', 
        default='./UCF101_Fishers_shuffled/train', 
        type=str, 
        help='Dir of shuffled training fisher data')
parser.add_argument('test_data_dir', 
        default='./UCF101_Fishers/test', 
        type=str, 
        help='Dir of testing fisher data')
parser.add_argument('train_list', 
        default='/home/xinqizhu/ucfTrainTestlist/trainlist01.txt', 
        type=str,
        help='Trainlist containing video and class')
parser.add_argument('test_list', 
        default='/home/xinqizhu/ucfTrainTestlist/testlist01.txt', 
        type=str,
        help='Testlist containing video and class')
parser.add_argument('per_class_num', 
        default=10000, 
        type=int, 
        help='Number of samples per class used to train, set large enough to use all')
parser.add_argument('PCA_dim', 
        default=None, 
        type=int, 
        help='Set PCA dim for train and test; set None to not use PCA')
parser.add_argument('save_dir', 
        default='./UCF101_Fishers', 
        type=str, 
        help='Dir to save results')
parser.add_argument('dataset', 
        default='UCF101', 
        type=str, 
        help='Dataset in UCF101 or Something')
parser.add_argument('--load_n_pca', 
        default=None, 
        type=str, 
        help='load normal pca file')
parser.add_argument('--load_s_pca', 
        default=None, 
        type=str, 
        help='load shuffled pca file')
parser.add_argument('--save_n_pca', 
        default=None, 
        type=str, 
        help='save normal pca file. when load_n_pca, it is not used')
parser.add_argument('--save_s_pca', 
        default=None, 
        type=str, 
        help='save shuffled pca file. when load_s_pca, it is not used')
parser.add_argument('--load_n_classifier', 
        default=None, 
        type=str, 
        help='load normal svm weights file')
parser.add_argument('--load_s_classifier', 
        default=None, 
        type=str, 
        help='load shuffled svm weights file')
parser.add_argument('--save_n_classifier', 
        default=None, 
        type=str, 
        help='save normal svm classifier file')
parser.add_argument('--save_s_classifier', 
        default=None, 
        type=str, 
        help='save shuffled svm classifier file')
args = parser.parse_args()


class_index_file = "./class_index.npz"
training_n_output = args.normal_train_data_dir
training_s_output = args.shuffled_train_data_dir
testing_output = args.test_data_dir

class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]


train_vid_class = classify_library.get_vid_class(args.train_list, index_class, args.dataset)
test_vid_class = classify_library.get_vid_class(args.test_list, index_class, args.dataset)

training_n = sorted([filename for filename in os.listdir(training_n_output) if filename.endswith('.fisher.npz')])
training_s = sorted([filename for filename in os.listdir(training_s_output) if filename.endswith('.fisher.npz')])
testing = sorted([filename for filename in os.listdir(testing_output) if filename.endswith('.fisher.npz')])

print(training_n[:5])
print(training_s[:5])
print(testing[:5])
print(train_vid_class.keys()[:5])
print('len testing:', len(testing))
training_n_dict = classify_library.toDict(training_n, train_vid_class)
training_s_dict = classify_library.toDict(training_s, train_vid_class)
testing_dict = classify_library.toDict(testing, test_vid_class)

# input('...')

#GET THE TRAINING AND TESTING DATA.


X_train_n_vids = classify_library.limited_input1(training_n_dict, args.per_class_num)
X_train_s_vids = classify_library.limited_input1(training_s_dict, args.per_class_num)
X_test_vids = classify_library.limited_input1(testing_dict, args.per_class_num)
# X_train_vids, X_test_vids = classify_library.limited_input(training_dict, testing_dict, 101, 24)
X_n_train, Y_n_train = classify_library.make_FV_matrix(X_train_n_vids, 
        training_n_output, class_index, train_vid_class)
X_s_train, Y_s_train = classify_library.make_FV_matrix(X_train_s_vids, 
        training_s_output, class_index, train_vid_class)
X_test, Y_test = classify_library.make_FV_matrix(X_test_vids, 
        testing_output, class_index, test_vid_class)

# pdb.set_trace()

training_n_PCA = classify_library.limited_input1(training_n_dict,1)
training_s_PCA = classify_library.limited_input1(training_s_dict,1)


if not args.PCA_dim:
    X_n_train_PCA = X_n_train.tolist()
    X_s_train_PCA = X_s_train.tolist()
    X_n_test_PCA = X_test.tolist()
    X_s_test_PCA = X_test.tolist()
else:
    # Experiments with PCA
    # pca_dim = 6000
    if args.load_n_pca == None:
        n_pca_dim = args.PCA_dim
        print('PCA to dim: ', str(n_pca_dim))
        n_pca = PCA(n_components=n_pca_dim)
        n_pca.fit(X_n_train)
        if not args.save_n_pca == None:
            pickle.dump(n_pca, open(args.save_n_pca, 'wb'))
    else:
        n_pca = pickle.load(open(args.load_n_pca, 'rb'))
    if args.load_s_pca == None:
        s_pca_dim = args.PCA_dim
        print('PCA to dim: ', str(s_pca_dim))
        s_pca = PCA(n_components=s_pca_dim)
        s_pca.fit(X_s_train)
        if not args.save_s_pca == None:
            pickle.dump(s_pca, open(args.save_s_pca, 'wb'))
    else:
        s_pca = pickle.load(open(args.load_n_pca, 'rb'))
    X_n_train_PCA = (n_pca.transform(X_n_train)).tolist()
    X_s_train_PCA = (s_pca.transform(X_s_train)).tolist()
    X_n_test_PCA = (n_pca.transform(X_test)).tolist()
    X_s_test_PCA = (s_pca.transform(X_test)).tolist()

print('Training SVM...')
# pdb.set_trace()
# prob  = svm_problem(Y_train, X_train_PCA)
# param = svm_parameter('-t 0 -c 100')
# mch = svm_train(prob, param)
# p_label, p_acc, p_val = svm_predict(Y_test, X_normal_test_PCA, mch)

# pdb.set_trace()

if args.load_n_classifier == None:
    n_estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
    n_estimator.fit(X_n_train_PCA, Y_n_train)
    if not args.save_n_classifier==None:
        pickle.dump(n_estimator, open(args.save_n_classifier, 'wb'))
else:
    n_estimator = pickle.load(open(args.load_n_classifier, 'rb'))
if args.load_s_classifier == None:
    s_estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
    s_estimator.fit(X_s_train_PCA, Y_s_train)
    if not args.save_s_classifier==None:
        pickle.dump(s_estimator, open(args.save_s_classifier, 'wb'))
else:
    s_estimator = pickle.load(open(args.load_s_classifier, 'rb'))
test_normal_scores = n_estimator.decision_function(X_n_test_PCA)
test_shuffled_scores = n_estimator.decision_function(X_s_test_PCA)
test_n_sm = [softmax(line) for line in test_normal_scores]
test_s_sm = [softmax(line) for line in test_shuffled_scores]

print('normal score:', test_normal_scores[0])
print('shuffled score:', test_shuffled_scores[0])

print('normal softmax:', test_n_sm[0])
print('shuffled softmax:', test_s_sm[0])

root_mse = [np.sqrt(mean_squared_error(test_n_sm[i], test_s_sm[i])) for i in range(len(test_n_sm))]
print(root_mse[:5])
print(np.mean(root_mse[:5]))
# dist = numpy.linalg.norm(a-b)
root_mse_mean = np.mean(root_mse)

# pred_normal_test = np.argmax(test_normal_scores, 1)+1
# pred_shuffled_test = np.argmax(test_shuffled_scores, 1)+1
pred_normal_test = np.argmax(test_n_sm, 1)+1
pred_shuffled_test = np.argmax(test_s_sm, 1)+1


acc_normal = float(np.sum(pred_normal_test==Y_test))/len(Y_test)
# print('Y_test:', Y_test[80:100])
acc_shuffled = float(np.sum(pred_shuffled_test==Y_test))/len(Y_test)
# print('Y_test:', Y_test[80:100])
print('normal acc:', acc_normal)
print('shuffled acc:', acc_shuffled)
print('root_mse_mean:', root_mse_mean)
# pdb.set_trace()
