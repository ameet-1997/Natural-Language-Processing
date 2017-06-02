import numpy as np
from scipy import sparse
from functions import load_from_arff, fill_bool_array, fit_classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import lil_matrix
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
import time

X_train = y_train = X_test = y_test = None

train_filename = "SLASHDOT-F-train.arff"
test_filename = "SLASHDOT-F-test.arff"

[X_train, y_train] = load_from_arff(filename=train_filename, load_sparse=True, labelcount=22)
[X_test, y_test] = load_from_arff(filename=test_filename, load_sparse=True, labelcount=22)

# The matrices are initially in lil_matrix format
# Converting them to compressed row matrix format

X_train = X_train.tocsr()
y_train = y_train.todense()
X_test = X_test.tocsr()
y_test = y_test.todense()

label_set = set([0, 3, 4, 7, 8, 9, 12, 15, 17, 19, 20, 21])
label_list = [[0, 3, 4, 7, 8, 9, 12, 15, 17, 19, 20, 21], [0, 12], [3, 4, 7, 9, 15, 17, 19, 21]]

# y_train = y_train[:,label_list[0]]
# y_test = y_test[:,label_list[0]]

start_time = time.process_time()
# classifier = LabelPowerset(RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1))
classifier = LabelPowerset(SGDClassifier(penalty='l2', alpha=0.01))
classifiers = [classifier for i in range(3)]
bool_shape = y_train.shape[0]
bool_array = [np.zeros(shape=bool_shape, dtype=bool) for i in range(3)]
bool_array[0] = np.ones(shape=bool_shape, dtype=bool)
bool_array[1] = np.zeros(shape=bool_shape, dtype=bool)
bool_array[2] = np.zeros(shape=bool_shape, dtype=bool)

bool_array = fill_bool_array(bool_array, label_list, y_train)

classifiers = fit_classifiers(classifiers, bool_array, label_list, X_train, y_train)
y_predicted = np.zeros(y_test.shape)
test_bool_array = np.zeros(shape=y_test.shape[0], dtype=bool)
for i in label_list[0]:
    test_bool_array = np.logical_or(np.array(test_bool_array), np.array(y_test[:,i] == 1).flatten())

X_test = X_test[test_bool_array,:]

y_predicted[:,[0,1,8,20]] = classifiers[0].predict(X_test)
test_bool_array = [None, None]
test_bool_array[0] = (y_predicted[:,0] == 1)
test_bool_array[1] = (y_predicted[:,1] == 1)
y_predicted[test_bool_array[0],label_list[1]] = classifiers[1].predict(X_test[test_bool_array[0],:])
y_predicted[test_bool_array[1],label_list[2]] = classifiers[2].predict(X_test[test_bool_array[1],:])

# y_predicted = classifier.predict(X_test)
total_time = time.process_time() - start_time

print("Total time taken is : "+str(total_time))

# print("Jaccard Similarity Score is : "+str(jaccard_similarity_score(y_test, y_predicted)))
# print("Hamming Loss is : "+str(hamming_loss(y_test, y_predicted)))