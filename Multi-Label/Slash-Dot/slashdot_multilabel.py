import numpy as np
from scipy import sparse
from functions import load_from_arff
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
from sklearn.naive_bayes import MultinomialNB
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


# parameters = {'n_estimators':range(1,16)}
# classifier = RandomForestClassifier()
# clf = GridSearchCV(classifier, parameters,scoring='f1_macro')
# clf.fit(X_train,y_train)

# print(clf.best_params_)


# Perform a grid search to find the number of estimators
# classifier = RandomForestClassifier(n_estimators=3)
# classifier.fit(X_train, y_train)
# y_predicted = classifier.predict(X_test)

# y_predicted = y_predicted.astype(int)

start_time = time.process_time()
classifier = LabelPowerset(RandomForestClassifier(random_state=0, n_estimators=10, min_samples_leaf=10, n_jobs=-1))
total_time = time.process_time() - start_time

print("Total time taken is : "+str(total_time))
# classifier = RandomForestClassifier(random_state=0, n_estimators=10, min_samples_leaf=10)
# classifier = BinaryRelevance(classifier = LinearSVC(), require_dense = [False, True])
# classifier = LabelPowerset(SGDClassifier(penalty='l2', alpha=0.01))
classifier.fit(X_train, y_train)
y_predicted = classifier.predict(X_test)

print("Jaccard Similarity Score is : "+str(jaccard_similarity_score(y_test, y_predicted)))
print("Hamming Loss is : "+str(hamming_loss(y_test, y_predicted)))
# print("F1_Similarity score is : "+str(f1_score(y_test,y_predicted,average='macro')))

# model_filename = "final_model.sav"
# pickle.dump(classifier, open(model_filename, 'wb'))