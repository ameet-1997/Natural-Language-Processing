import numpy as np
from sklearn.datasets import fetch_rcv1
from global_variables import is_leaf_topic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_similarity_score

# Fetch the training dataset
train_data = fetch_rcv1(subset='train')

# Convert the scipy sparse matrix to a dense version usable by sklearn's functions
train_data.target = train_data.target.todense()
is_leaf_topic = np.array(is_leaf_topic)

# Subset the data to choose documents part of leaf nodes
train_data.target = train_data.target[:,is_leaf_topic]

# Train the classifier with the training data
classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(train_data.data, train_data.target)

# Fetch the test data
test_data = fetch_rcv1(subset='test',random_state=42,shuffle=True)
test_data.data = test_data.data[0:1000,:]
test_data.target = test_data.target[0:1000,:]
test_data.target = test_data.target[:,is_leaf_topic]
test_data.target = test_data.target.todense()



test_predict = classifier.predict(test_data.data)

print("The Jaccard Similiarity Score is : "+str(jaccard_similarity_score(test_data.target, test_predict)))