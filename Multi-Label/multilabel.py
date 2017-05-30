from scipy import sparse
from functions import load_from_arff
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import lil_matrix

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

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)
y_predicted = classifier.predict(X_test)

f = open("values.txt","w")
for i in range(len(y_test)):
    s = ""
    for j in range(22):
        s = s+" "+str(y_predicted[i,j])
    s = s+"\n"
    f.write(s)
f.close()
