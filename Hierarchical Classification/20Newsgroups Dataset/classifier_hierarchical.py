import os
import copy     # For performing a deepcopy of the dataframe
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer 	# Using inverse document frequency to filter the noise
from sklearn.pipeline import Pipeline
import sys
from sklearn import metrics
sys.path.insert(0,"C:\\Users\\A6000401\\Desktop\\NLP\\20 Newsgroup Datasets\\20news-bydate\\20news-bydate-test")
from get_test_data import test_data_function
from get_test_data import stop
from get_test_data import topic_mapping     # Import a dictionary to map topic names to integer labels
import time
start_time = time.time()

# Function declarations

def modify_dataframe(x):
    if x in [3,4,5,6]:
        return 1
    elif x in [9,10]:
        return 2
    elif x in [7,8]:
        return 3
    elif x in [17,18]:
        return 4
    else:
        return 5

def modify_data_children(data_children):
    data = []
    data.append(data_children[0].loc[data_children[0]['class'].isin([3,4,5,6])])    # Computer
    data.append(data_children[1].loc[data_children[1]['class'].isin([9,10])])       # Sports
    data.append(data_children[2].loc[data_children[2]['class'].isin([7,8])])        # Automotive
    data.append(data_children[3].loc[data_children[3]['class'].isin([17,18])])      # Politics
    return data

def predict_leaf(root_predictions):
    till = len(root_predictions)
    for i in range(till):
        if root_predictions[i] == 1:
            root_predictions[i] = pipeline_children[0].predict([test_data_frame['text'][i]])
        elif root_predictions[i] == 2:
            root_predictions[i] = pipeline_children[1].predict([test_data_frame['text'][i]])
        elif root_predictions[i] == 3:
            root_predictions[i] = pipeline_children[2].predict([test_data_frame['text'][i]])
        elif root_predictions[i] == 4:
            root_predictions[i] = pipeline_children[3].predict([test_data_frame['text'][i]])
    return root_predictions

# End of function declarations

rows = []  # Contains the text of the article and the class that it belongs to
index = []  # Unique index, here the file name

# Walk through all the files and store the content in a data frame

for (root, dirs, files) in os.walk('.', topdown=True):
    for name in dirs:

        # For all the sub-directories in the root folder, if the directory contains articles

        if str(name) in topic_mapping:
            current_path = str(os.path.join(root, name))
            files1 = os.listdir(current_path)
            for file_name in files1:  # For all the files in the root directory
                f1 = open(current_path + str('/') + str(file_name), 'r')
                content = f1.readlines()
                content = ' '.join(content)  # Joins the contents of the list into one single string separated by a space
                rows.append({'text': content,
                            'class': topic_mapping[str(name)]})  # Based on the directory, assign the class value
                index.append(str(file_name))  # Use the file name as the index of the entry

                f1.close()

# Create a dataframe with text as one column and class as the other column
data_frame = DataFrame(rows, index=index)

# Subset only the data that we want
data_frame = data_frame.loc[data_frame['class'].isin([3,4,5,6,7,8,9,10,17,18])]

data_frame_modified = copy.deepcopy(data_frame)    # To be used for first-level fitting
# Assigning new labels, {"Computer":1, "Sports":2, "Automobile":3, "Politics:4"} for first level classification

data_frame_modified['class'] = data_frame_modified['class'].apply(modify_dataframe)
print("Length of the training data : ")
print(len(data_frame_modified))

# Adding few common words that are frequent in this dataset, but do not contribute to class resolution
# nltk.corpora stopwords not working currently

stop.extend((str('subject'), str('from'), str('organization'), str('organisation')))

# Using pipeline_root to fit data to model and then convert it to tf-idf counts
pipeline_root = Pipeline([
    ('count_vectorizer',   CountVectorizer(stop_words=stop, ngram_range=(1,  2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',     SGDClassifier())
])

# Fit the main pipeline with 
pipeline_root.fit(data_frame_modified['text'].values, data_frame_modified['class'].values)

# Creates 4 pipelines, one for each child node
pipeline_children = [Pipeline([
    ('count_vectorizer',   CountVectorizer(stop_words=stop, ngram_range=(1,  2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',     SGDClassifier())
]) for i in range(4)]

# Create data copies for all the children
data_children = [data_frame for i in range(4)]
data_children = modify_data_children(data_children)

# Fit Models for each of the children
# Local Classifier Per Parent Approach

pipeline_children[0].fit(data_children[0]['text'].values, data_children[0]['class'].values)
pipeline_children[1].fit(data_children[1]['text'].values, data_children[1]['class'].values)
pipeline_children[2].fit(data_children[2]['text'].values, data_children[2]['class'].values)
pipeline_children[3].fit(data_children[3]['text'].values, data_children[3]['class'].values)


test_data_frame = test_data_function()   # Test data is obtained as a data frame
# Get the parent predictions
root_predictions = pipeline_root.predict(test_data_frame["text"])

predictions = predict_leaf(root_predictions)

# Test the prediction
correct_answers = test_data_frame["class"]
accuracy = metrics.accuracy_score(correct_answers, predictions, normalize=True)

print("The Accuracy is : ")
print(accuracy*100)
print("The F-Score is : ")
print(metrics.f1_score(correct_answers, predictions, average='macro'))
print("Total number of articles in test set it : ")
print(len(predictions))

print("--- %s seconds ---" % (time.time() - start_time))