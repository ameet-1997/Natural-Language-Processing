from textblob import TextBlob
from pandas import read_csv, read_excel
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster.bicluster import SpectralBiclustering

# Read the excel file and store in a dataframe
df = read_excel("smaller_version.xlsx",encoding='latin-1')

# Dictionary stores all nouns present in the reviews
all_nouns = {}

for sent in df["Reviews"]:
	# Convert the sentence into a textblob
	try:
		blob = TextBlob(sent)
	except:
		continue

	for txt in blob.sentences:
		txt = str(txt)
		txt = nltk.word_tokenize(txt)
		# Perform part-of-speech tagging for each sentence seperately
		txt = nltk.pos_tag(txt)

		# For each word, if it is a noun, store it in a dictionary
		for word in txt:
			if not word[0].isalpha():
				continue
			if word[1] == 'NN' or word[1] == 'NNS':
				if word[0].lower() in all_nouns:
					all_nouns[word[0].lower()] += 1
				else:
					all_nouns[word[0].lower()] = 1


# Extract nouns which have a count of more than 'threshold_nouns'
# Threshold is set as 1% of the total number of reviews used

# Change the threshold ---------- ----------------- -----------------        -------------------
threshold_nouns = df["Reviews"].size/30
main_nouns = dict((k, v) for k, v in all_nouns.items() if v >= threshold_nouns)
# print(main_nouns)

# Mapping nouns to numbers so that a similarity score matrix can be built

# Sort the keys
sorted_keys = sorted(main_nouns)
# Store the mapping in a dictionary
word_to_number = {}
# Store the inverse mapping
inverse_word_to_number = {}
# Map the words to numbers in alphabetical order
counter = 0
for word in sorted_keys:
	word_to_number[word] = counter
	inverse_word_to_number[counter] = word
	counter += 1

number_of_attributes = len(word_to_number)

# Create a matrix that will store the similarity scores
similarity_matrix = np.ones((number_of_attributes, number_of_attributes), dtype=np.float32)

f = open("similarity_values.txt","w")

# For all word pairs find semantic similarity
for i in range(number_of_attributes):
	for j in range(number_of_attributes):
		# Generate synsets for all the words
		syn1 = wn.synsets(inverse_word_to_number[i])
		syn2 = wn.synsets(inverse_word_to_number[j])
		maximum_score = 0
		# Out of all the synset matchings, choose the one with the maximum score
		for syn_i in syn1:
			for syn_j in syn2:
				# Calculate the similarity
				try:
					maximum_score = max(maximum_score, syn_i.path_similarity(syn_j))
				except:
					continue

		similarity_matrix[i,j] = 1 - maximum_score
		f.write(str(maximum_score)+" ")
	f.write("\n")



clustering = SpectralBiclustering()
clustering.fit(similarity_matrix)

contents_of_cluster = ["" for i in range(np.unique(clustering.row_labels_).shape[0])]

for i in range(clustering.row_labels_.shape[0]):
	contents_of_cluster[clustering.row_labels_[i]] = contents_of_cluster[clustering.row_labels_[i]] + " " + str(inverse_word_to_number[i])

for content in contents_of_cluster:
	print(content)
