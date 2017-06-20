"""
Creates a list which stores the number od documents against the labels
"""

from sklearn.datasets import fetch_rcv1

training_dataset = fetch_rcv1(subset='train')
counts = training_dataset.target.todense()

numbers_of_documents = []

for i in range(len(training_dataset.target_names)):
	number_of_documents.append(counts[:,i].flatten().sum())

f = open('labels_vs_count.txt','w')

for i in range(len(training_dataset.target_names)):
	f.write(training_dataset.target_names[i] + " : " + str(number_of_documents[i])+"\n")
