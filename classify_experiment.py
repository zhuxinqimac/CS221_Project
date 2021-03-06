
"""
Script to train a basic action classification system.

Trains a One vs. Rest SVM classifier on the fisher vector video outputs.
This script is used to experimentally test different parameter settings for the SVMs.

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
from svmutil import *
import sklearn.metrics as metrics
import classify_library
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import argparse

parser = argparse.ArgumentParser()


parser.add_argument('train_data_dir', 
        default='./UCF101_Fishers/train', 
        type=str, 
        help='Dir of training fisher data')
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
testing_output = args.test_data_dir

class_index_file_loaded = np.load(class_index_file)
class_index = class_index_file_loaded['class_index'][()]
index_class = class_index_file_loaded['index_class'][()]


train_vid_class = classify_library.get_vid_class(args.train_list, index_class, args.dataset)
test_vid_class = classify_library.get_vid_class(args.test_list, index_class, args.dataset)

training = [filename for filename in os.listdir(training_output) if filename.endswith('.fisher.npz')]
testing = [filename for filename in os.listdir(testing_output) if filename.endswith('.fisher.npz')]

print(training[:5])
print(testing[:5])
print(train_vid_class.keys()[:5])
training_dict = classify_library.toDict(training, train_vid_class)
testing_dict = classify_library.toDict(testing, test_vid_class)


#GET THE TRAINING AND TESTING DATA.


X_train_vids = classify_library.limited_input1(training_dict, args.per_class_num)
X_test_vids = classify_library.limited_input1(testing_dict, args.per_class_num)
# X_train_vids, X_test_vids = classify_library.limited_input(training_dict, testing_dict, 101, 24)
X_train, Y_train = classify_library.make_FV_matrix(X_train_vids,training_output, class_index, train_vid_class)
X_test, Y_test = classify_library.make_FV_matrix(X_test_vids,testing_output, class_index, test_vid_class)

# pdb.set_trace()

training_PCA = classify_library.limited_input1(training_dict,1)


if not args.PCA_dim:
    X_train_PCA = X_train.tolist()
    X_test_PCA = X_test.tolist()
else:
    # Experiments with PCA
    # pca_dim = 6000
    if args.load_pca == None:
        pca_dim = args.PCA_dim
        print('PCA to dim: ', str(pca_dim))
        pca = PCA(n_components=pca_dim)
        pca.fit(X_train)
        if not args.save_pca == None:
            pca_file = args.pca_file
            pickle.dump(pca, open(pca_file, 'wb'))
    else:
        pca = pickle.load(open(args.load_pca, 'rb'))
    X_train_PCA = (pca.transform(X_train)).tolist()
    X_test_PCA = (pca.transform(X_test)).tolist()

print('Training SVM...')
# pdb.set_trace()
# prob  = svm_problem(Y_train, X_train_PCA)
# param = svm_parameter('-t 0 -c 100')
# mch = svm_train(prob, param)
# p_label, p_acc, p_val = svm_predict(Y_test, X_test_PCA, mch)

# pdb.set_trace()

estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
if load_classifier == None:
    # classifier = estimator.fit(X_train_PCA, Y_train)
    estimator.fit(X_train_PCA, Y_train)
    if not save_classifier==None:
        # pickle.dump(classifier, open(args.save_classifier, 'wb'))
        pickle.dump(estimator, open(args.save_classifier, 'wb'))
else:
    # classifier = pickle.load(open(args.load_classifier, 'rb'))
    estimator = pickle.load(open(args.load_classifier, 'rb'))
test_scores = estimator.decision_function(X_test_PCA)

test_scores_path = os.path.join(args.save_dir, 'test_scores')
# np.savez(test_scores_path, scores=test_scores)
with open(test_scores_path, 'w') as f:
    for line in test_scores:
        f.write(str(line)+'\n')

pred_test = np.argmax(test_scores, 1)+1

test_pred_path = os.path.join(args.save_dir, 'test_pred')
# np.savez(test_pred_path, pred=pred_test)
with open(test_pred_path, 'w') as f:
    for line in pred_test:
        f.write(str(line)+'\n')

acc = float(np.sum(pred_test==Y_test))/len(Y_test)
with open(os.path.join(args.save_dir, 'test_acc.txt'), 'w') as f:
    f.write(str(acc))
print('Acc: ', str(acc))
# pdb.set_trace()

# metrics = classify_library.metric_scores(classifier, X_test_PCA, Y_test, verbose=True)
# print metrics


do_learning_curve = False
if do_learning_curve:
    X_full = np.vstack([X_train_PCA, X_test_PCA])
    Y_full = np.hstack([Y_train, Y_test])
    title= "Learning Curves (Linear SVM, C: %d, loss: %s, penalty: %s, PCA dim: %d)" % (100,'l1','l2',pca_dim)
    cv = cross_validation.ShuffleSplit(X_full.shape[0], n_iter=4,test_size=0.2, random_state=0)
    estimator = OneVsRestClassifier(LinearSVC(random_state=0, C=100, loss='l1', penalty='l2'))
    plot_learning_curve(estimator, title, X_full, Y_full, (0.7, 1.01), cv=cv, n_jobs=1)
    plt.show()

