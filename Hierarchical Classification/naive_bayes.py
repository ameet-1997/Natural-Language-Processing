"""
This model implements the Naive Bayes classifier with
one classifier per parent node.
The hierarchy is assumed to be given in a file in the 
format described below. Parent Node Child Node
The first node is assumed to represent the root
The labels are assumed to not have any space in 
their names.

20Newsgroup Dataset is being used

It is assumed that the document can only be part of leaves

Author : Ameet S Deshpande
"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from global_variables import stop, topic_mapping, inverse_mapping, leaf_to_topic, inverse_leaf_to_topic, cats
from functions import build_hierarchy, train_classifiers, build_classifier_map, predict_class
import time

start = time.clock()

train_dataset = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

# Adjacency list represents the hieararchy tree
# node_int_map maps the node label to the adjacency list index
# node_int_inverse_map represents the inverse of node_int_map
# parent_nos represents the number of nodes which are not leaves
[adjacency_list, node_int_map, node_int_inverse_map, parent_nos] = build_hierarchy()

count_vectorizer = CountVectorizer(stop_words=stop, ngram_range=(1,  2))
tfidf_transformer = TfidfTransformer()
features = count_vectorizer.fit_transform(train_dataset.data)
features = tfidf_transformer.fit_transform(features)
print("----Built Features----")

classifier_map = build_classifier_map(adjacency_list)
# Construct one clissifier for each parent node
classifiers = [MultinomialNB() for i in range(parent_nos)]
# 0 represents the root
garbage = train_classifiers(classifiers, adjacency_list, 0, features, np.array(train_dataset.target), node_int_inverse_map, leaf_to_topic, classifier_map)
print("----Training Done----")

test_dataset = fetch_20newsgroups(subset='test', categories=cats, remove=('headers', 'footers', 'quotes'))
actual_answers = test_dataset.target
documents = tfidf_transformer.transform(count_vectorizer.transform(test_dataset.data))
predictions = predict_class(documents, classifiers, classifier_map, inverse_leaf_to_topic, node_int_map, count_vectorizer, tfidf_transformer)

print("----Testing Done----")

print("The Accuracy is : "+str(metrics.accuracy_score(actual_answers, predictions, normalize=True)*100))
print("The F-Score is : "+str(metrics.f1_score(actual_answers, predictions, average='macro')))
print("Total number of articles in test set it : "+str(len(predictions)))

print("Time Elapsed : "+str(time.clock()-start))
