
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

parser.add_argument('train_data_dir', 
        default='./UCF101_Fishers/train', 
        type=str, 
        help='Dir of training fisher data')
parser.add_argument('normal_test_data_dir', 
        default='./UCF101_Fishers/test', 
        type=str, 
        help='Dir of testing fisher data')
parser.add_argument('shuffled_test_data_dir', 
        default='./UCF101_Fishers_shuffled/test', 
        type=str, 
        help='Dir of testing shuffled fisher data')
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
parser.add_argument('--load_pca', 
        default=None, 
        type=str, 
        help='load pca file')
parser.add_argument('--save_pca', 
        default=None, 
        type=str, 
        help='save pca file. when load_pca, it is not used')
parser.add_argument('--load_classifier', 
        default=None, 
        type=str, 
        help='load svm weights file')
parser.add_argument('--save_classifier', 
        default=None, 
        type=str, 
        help='save svm classifier file')
args = parser.parse_args()


class_index_file = "./class_index.npz"
training_output = args.train_data_dir
normal_testing_output = args.normal_test_data_dir
shuffled_testing_output = args.shuffled_test_data_dir

class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]


train_vid_class = classify_library.get_vid_class(args.train_list, index_class, args.dataset)
test_vid_class = classify_library.get_vid_class(args.test_list, index_class, args.dataset)

training = [filename for filename in os.listdir(training_output) if filename.endswith('.fisher.npz')]
normal_testing = sorted([filename for filename in os.listdir(normal_testing_output) if filename.endswith('.fisher.npz')])
shuffled_testing = sorted([filename for filename in os.listdir(shuffled_testing_output) if filename.endswith('.fisher.npz')])

print(training[:5])
print(normal_testing[:5])
print(shuffled_testing[:5])
print(train_vid_class.keys()[:5])
for i in range(len(normal_testing)):
    if normal_testing[i] != shuffled_testing[i]:
        print('different element!')
print('len normal_testing:', len(normal_testing))
print('len shuffled_testing:', len(shuffled_testing))
training_dict = classify_library.toDict(training, train_vid_class)
normal_testing_dict = classify_library.toDict(normal_testing, test_vid_class)
shuffled_testing_dict = classify_library.toDict(shuffled_testing, test_vid_class)

# input('...')

#GET THE TRAINING AND TESTING DATA.


X_train_vids = classify_library.limited_input1(training_dict, args.per_class_num)
X_normal_test_vids = classify_library.limited_input1(normal_testing_dict, args.per_class_num)
X_shuffled_test_vids = classify_library.limited_input1(shuffled_testing_dict, args.per_class_num)
# X_train_vids, X_test_vids = classify_library.limited_input(training_dict, testing_dict, 101, 24)
X_train, Y_train = classify_library.make_FV_matrix(X_train_vids, 
        training_output, class_index, train_vid_class)
X_normal_test, Y_normal_test = classify_library.make_FV_matrix(X_normal_test_vids, 
        normal_testing_output, class_index, test_vid_class)
X_shuffled_test, Y_shuffled_test = classify_library.make_FV_matrix(X_shuffled_test_vids, 
        shuffled_testing_output, class_index, test_vid_class)

# pdb.set_trace()

training_PCA = classify_library.limited_input1(training_dict,1)


if not args.PCA_dim:
    X_train_PCA = X_train.tolist()
    X_normal_test_PCA = X_normal_test.tolist()
    X_shuffled_test_PCA = X_shuffled_test.tolist()
else:
    # Experiments with PCA
    # pca_dim = 6000
    if args.load_pca == None:
        pca_dim = args.PCA_dim
        print('PCA to dim: ', str(pca_dim))
        pca = PCA(n_components=pca_dim)
        pca.fit(X_train)
        if not args.save_pca == None:
            pickle.dump(pca, open(args.save_pca, 'wb'))
    else:
        pca = pickle.load(open(args.load_pca, 'rb'))
    X_train_PCA = (pca.transform(X_train)).tolist()
    X_normal_test_PCA = (pca.transform(X_normal_test)).tolist()
    X_shuffled_test_PCA = (pca.transform(X_shuffled_test)).tolist()

print('Training SVM...')
# pdb.set_trace()
# prob  = svm_problem(Y_train, X_train_PCA)
# param = svm_parameter('-t 0 -c 100')
# mch = svm_train(prob, param)
# p_label, p_acc, p_val = svm_predict(Y_test, X_normal_test_PCA, mch)

# pdb.set_trace()

if args.load_classifier == None:
    estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
    estimator.fit(X_train_PCA, Y_train)
    if not args.save_classifier==None:
        pickle.dump(estimator, open(args.save_classifier, 'wb'))
else:
    estimator = pickle.load(open(args.load_classifier, 'rb'))
test_normal_scores = estimator.decision_function(X_normal_test_PCA)
test_n_sm = [softmax(line) for line in test_normal_scores]
test_shuffled_scores = estimator.decision_function(X_shuffled_test_PCA)
test_s_sm = [softmax(line) for line in test_shuffled_scores]

print('normal score:', test_normal_scores[0])
print('shuffled score:', test_shuffled_scores[0])

print('normal softmax:', test_n_sm[0])
print('shuffled softmax:', test_s_sm[0])

print('minus:', test_n_sm[0]-test_s_sm[0])
root_mse = [np.sqrt(mean_squared_error(test_n_sm[i], test_s_sm[i])) for i in range(len(test_n_sm))]
print(root_mse[:5])
print(np.mean(root_mse[:5]))
# dist = numpy.linalg.norm(a-b)
root_mse_mean = np.mean(root_mse)

# pred_normal_test = np.argmax(test_normal_scores, 1)+1
# pred_shuffled_test = np.argmax(test_shuffled_scores, 1)+1
pred_normal_test = np.argmax(test_n_sm, 1)+1
pred_shuffled_test = np.argmax(test_s_sm, 1)+1


acc_normal = float(np.sum(pred_normal_test==Y_normal_test))/len(Y_normal_test)
print('Y_normal_test:', Y_normal_test[:100])
acc_shuffled = float(np.sum(pred_shuffled_test==Y_shuffled_test))/len(Y_shuffled_test)
print('Y_shuffled_test:', Y_shuffled_test[:100])
print('normal acc:', acc_normal)
print('shuffled acc:', acc_shuffled)
print('root_mse_mean:', root_mse_mean)
# pdb.set_trace()
