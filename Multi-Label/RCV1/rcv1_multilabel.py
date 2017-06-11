import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import jaccard_similarity_score
from copy import deepcopy


def evaluate_score(train_dataset, test_dataset, number_of_labels) : 
	# train = fetch_rcv1(subset='train', shuffle=True, random_state=42)
	# number_of_labels = 10
	train = deepcopy(train_dataset)
	test = deepcopy(test_dataset)

	train.target = train.target[:,range(number_of_labels)]
	bool_array = np.zeros(shape=train.target.shape[0], dtype=bool)

	for i in range(number_of_labels):
		bool_array = np.logical_or(bool_array, np.array(((train.target[:,i]==1).todense())).flatten())

	train.data = train.data[bool_array,:]
	train.target = train.target[bool_array,:]

	print("Total number of train documents : "+str(train.target.shape[0]))

	classifier = RandomForestClassifier(n_estimators=10)
	classifier.fit(train.data, train.target.todense())

	test = fetch_rcv1(subset='test', shuffle=True, random_state=42)
	test.data = test.data[:30000,:]
	test.target = test.target[:30000,:]
	test.target = test.target[:,range(number_of_labels)]

	bool_array = np.zeros(shape=test.target.shape[0], dtype=bool)

	for i in range(number_of_labels):
		bool_array = np.logical_or(bool_array, np.array(((test.target[:,i]==1).todense())).flatten())

	test.data = test.data[bool_array,:]
	test.target = test.target[bool_array,:]

	print("Total number of test documents : "+str(test.target.shape[0]))

	predicted = classifier.predict(test.data)

	# print("Jaccard Similarity Score is : "+str(jaccard_similarity_score(test.target, predicted)))
	return jaccard_similarity_score(test.target, predicted)
