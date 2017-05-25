from global_variables import stop, topic_mapping, inverse_mapping, leaf_to_topic, inverse_leaf_to_topic
import numpy as np

# Reads the hierarchy file and builds adjacency list
def build_hierarchy():
	# Read the lines from the file
	file_pointer = open("hierarchical_structure.txt", "r")
	edges = file_pointer.readlines()
	file_pointer.close()

	# Variables to return
	node_int_map = {} 			# Maps the label of the node to an integer label which represents the node in the list
	node_int_inverse_map = {}	# Represents the inverse mapping of the previous dictionary
	adjacency_list = [] 		# Represents the tree that has been given as the input
	counter = 0
	parent_nos = 0

	# Go through all the parent-child relationship
	for edge in edges:
		edge = edge.strip('\n').split()		# edge now contains a list with parent, child as elements
		for i in range(2) :	
			if int(edge[i]) not in node_int_map.keys() : 	# If a integer has not been assigned to the node yet
				node_int_map[int(edge[i])] = counter
				node_int_inverse_map[counter] = int(edge[i])
				counter += 1
				adjacency_list.append([]) 	# Append an empty list to the adjacency list
		adjacency_list[node_int_map[int(edge[0])]].append(node_int_map[int(edge[1])])

	for node in adjacency_list : 
		if node != [] :
			parent_nos += 1

	# Return the list of the three items
	return [adjacency_list, node_int_map, node_int_inverse_map, parent_nos]

def train_classifiers(classifiers, adjacency_list, node, features, train_dataset_target, node_int_inverse_map, leaf_to_topic, classifier_map) :
	if not adjacency_list[node] :
		documents = node
		documents = leaf_to_topic[node_int_inverse_map[node]]
		boolean_array = (train_dataset_target == documents)		# Vectorizing the code
		return boolean_array
	else :
		boolean_array = np.zeros(shape=len(train_dataset_target), dtype=bool)
		local_target = np.ones(shape=len(train_dataset_target), dtype=int)
		for child in adjacency_list[node] : 
			temp_array = train_classifiers(classifiers, adjacency_list, child, features, train_dataset_target, node_int_inverse_map, leaf_to_topic, classifier_map)
			local_target[temp_array] = child
			boolean_array = np.logical_or(boolean_array, temp_array)
		local_features = features[boolean_array,:]
		local_target = local_target[boolean_array]
		print("Node Number : "+str(node)+" Number of Training examples : "+str(len(local_target)))
		# zz = {}
		# for i in local_target : 
		# 	zz[i] = 1
		# print("The children nodes are : ")
		# print(zz.keys())
		classifiers[classifier_map[node]].fit(local_features, local_target)
		return boolean_array


def build_classifier_map(adjacency_list) :
	classifier_map = {}
	counter = 0
	till = len(adjacency_list)
	for i in range(till):
		if adjacency_list[i] != [] :
			classifier_map[i] = counter
			counter += 1
	return classifier_map

def predict_class(documents, classifiers, classifier_map, leaf_to_topic, node_int_inverse_map, count_vectorizer, tfidf_transformer):
	till = documents.shape[0]
	final_answer = classifiers[0].predict(documents)
	for i in range(till) : 
		current_class = final_answer[i]
		document = documents[i]
		while current_class in classifier_map:
			current_class = classifiers[classifier_map[current_class]].predict(document)[0]
		# 	print("Current Class is : "+str(current_class))
		# print("--------------")
		final_answer[i] = leaf_to_topic[node_int_inverse_map[current_class]]
	return final_answer

# def predict_class(documents, classifiers, classifier_map, leaf_to_topic, node_int_inverse_map, count_vectorizer, tfidf_transformer):
# 	till = documents.shape[0]
# 	final_answer = []
# 	for i in range(till) : 
# 		current_class = 0
# 		document = documents[i]
# 		while current_class in classifier_map:
# 			current_class = classifiers[classifier_map[current_class]].predict(document)[0]
# 		# 	print("Current Class is : "+str(current_class))
# 		# print("--------------")
# 		final_answer.append(leaf_to_topic[node_int_inverse_map[current_class]])
# 	return final_answer
