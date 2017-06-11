import numpy as np
from sklearn.datasets import fetch_rcv1
from rcv1_multilabel import evaluate_score

train_dataset = fetch_rcv1(subset='train', shuffle=True, random_state=42)
test_dataset = fetch_rcv1(subset='test', shuffle=True, random_state=42)
test_dataset.data = test_dataset.data[:40000,:] 
test_dataset.target = test_dataset.target[:40000,:]

number_of_labels = []
jaccard_score = []

for i in range(2,100):
	jaccard_score.append(evaluate_score(train_dataset, test_dataset, i))
	number_of_labels.append(i)

# number_of_labels
# jaccard_score

f = open("values.txt","w")

for i in range(len(number_of_labels)):
	f.write(str(number_of_labels[i])+" : "+str(jaccard_score[i])+"\n")
