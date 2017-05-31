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
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import time

start_time = time.process_time()
train_filename = "wise2014-train.arff"
# test_filename = "SLASHDOT-F-test.arff"

labelcount = 203

[X, y] = load_from_arff(filename=train_filename, load_sparse=True, labelcount=labelcount, endian="little")
# [X_test, y_test] = load_from_arff(filename=test_filename, load_sparse=True, labelcount=labelcount)

X = X[:,1:]

print("Loading done : "+str(time.process_time() - start_time))

X = X.tocsr()
y = y.todense()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("Splitting done : "+str(time.process_time() - start_time))

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train,y_train)
y_predicted = classifier.predict(X_test)
y_predicted = y_predicted.astype(int)

# classifier = ClassifierChain(RandomForestClassifier(n_estimators=10))
# classifier.fit(X_train, y_train)
# y_predicted = classifier.predict(X_test)

print("Jaccard Similarity Score is : "+str(jaccard_similarity_score(y_test, y_predicted)))
