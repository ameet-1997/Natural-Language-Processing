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
import copy

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

# label_set = set([0, 3, 4, 7, 8, 9, 12, 15, 17, 19, 20, 21])
label_list = [[0, 3, 4, 7, 8, 9, 12, 15, 17, 19, 20, 21], [0, 12], [3, 4, 7, 9, 15, 17, 19, 21]]

# y_train = y_train[:,label_list[0]]
# y_test = y_test[:,label_list[0]]

start_time = time.time()
# Initialize the classifiers, one per parent
# classifier = LabelPowerset(RandomForestClassifier(random_state=0, n_estimators=10, n_jobs=-1))
classifier = LabelPowerset(SGDClassifier(penalty='l2', alpha=0.01))
classifiers = []
for i in range(3):
	classifiers.append(copy.deepcopy(classifier))

# Create boolean array for choosing relevant documents for each parent classifer
bool_shape = y_train.shape[0]
bool_array = [np.zeros(shape=bool_shape, dtype=bool) for i in range(3)]
# bool_array[0] = np.ones(shape=bool_shape, dtype=bool)
# bool_array[1] = np.zeros(shape=bool_shape, dtype=bool)
# bool_array[2] = np.zeros(shape=bool_shape, dtype=bool)

# Populate the boolean arrays
bool_array = fill_bool_array(bool_array, label_list, y_train)

# Fit the classifiers
classifiers = fit_classifiers(classifiers, bool_array, label_list, X_train, y_train)

# Choose only those test documents that belong to one of the chosen categories
test_bool_array = np.zeros(shape=y_test.shape[0], dtype=bool)
for i in label_list[0]:
    test_bool_array = np.logical_or(test_bool_array, np.array(y_test[:,i] == 1).flatten())

X_test = X_test[test_bool_array,:]
y_test = y_test[test_bool_array,:]

# y_predicted will store the predicted results
y_predicted = np.zeros(y_test.shape)
# print(type(y_predicted))
# print(y_predicted.shape)
# print(X_test.shape)

# answer = classifiers[0].predict(X_test)
# print("The shape of answer is : "+str(answer.shape))
# print("The type of answer is : "+str(type(answer)))
# answer = answer.todense()
# print("The type of answer after conversion is : "+str(type(answer)))

y_predicted[:,[0,1,8,20]] = classifiers[0].predict(X_test).todense()
test_bool_array = [None, None]
test_bool_array[0] = (y_predicted[:,0] == 1)
test_bool_array[1] = (y_predicted[:,1] == 1)
print("Shape of boolean array : "+str(test_bool_array[0].shape))
temp = y_predicted[test_bool_array[0],:][:,label_list[1]]
# print("Testing shapes :: "+str(y_predicted[test_bool_array[0],label_list[1]].shape))#+" :: "+str(classifiers[1].predict(X_test[test_bool_array[0],:]).todense().shape))
# y_predicted[test_bool_array[0],label_list[1]] = classifiers[1].predict(X_test[test_bool_array[0],:]).todense()
# y_predicted[test_bool_array[1],label_list[2]] = classifiers[2].predict(X_test[test_bool_array[1],:]).todense()

# y_predicted = classifier.predict(X_test)
total_time = time.time() - start_time

print("Total time taken is : "+str(total_time))

# print("Jaccard Similarity Score is : "+str(jaccard_similarity_score(y_test, y_predicted)))
# print("Hamming Loss is : "+str(hamming_loss(y_test, y_predicted))) 
