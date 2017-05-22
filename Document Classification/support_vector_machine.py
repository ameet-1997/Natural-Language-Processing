import os
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.corpus import stopwords
#from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer 	# Using inverse document frequency to filter the noise
from sklearn.pipeline import Pipeline
import sys
from sklearn import metrics
sys.path.insert(0,"C:\\Users\\A6000401\\Desktop\\NLP\\20 Newsgroup Datasets\\20news-bydate\\20news-bydate-test")
from get_test_data import test_data_function

# Create a dictionary to map topic names to integer labels

topic_mapping = {
    'alt.atheism': 1,
    'comp.graphics': 2,
    'comp.os.ms-windows.misc': 3,
    'comp.sys.ibm.pc.hardware': 4,
    'comp.sys.mac.hardware': 5,
    'comp.windows.x': 6,
    'rec.autos': 7,
    'rec.motorcycles': 8,
    'rec.sport.baseball': 9,
    'rec.sport.hockey': 10,
    'sci.crypt': 11,
    'sci.electronics': 12,
    'sci.med': 13,
    'sci.space': 14,
    'misc.forsale': 15,
    'talk.politics.misc': 16,
    'talk.politics.guns': 17,
    'talk.politics.mideast': 18,
    'talk.religion.misc': 19,
    'soc.religion.christian': 20,
    }

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

print(len(data_frame))

# Adding few common words that are frequent in this dataset, but do not contribute to class resolution
stop = stopwords.words('english')
stop = list(stop)
stop.extend((str('subject'), str('from'), str('organization'
            ), str('organisation')))

# Using pipeline to fit data to model and then convert it to tf-idf counts
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1,  2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',     SGDClassifier())
])

pipeline.fit(data_frame['text'].values, data_frame['class'].values)

test_data_frame = test_data_function()   # Test data is obtained as a data frame

predictions = pipeline.predict(test_data_frame["text"])
correct_answers = test_data_frame["class"]
accuracy = metrics.accuracy_score(correct_answers, predictions, normalize=True)

print("The Accuracy is : ")
print(accuracy*100)
print("The F-Score is : ")
print(metrics.f1_score(correct_answers, predictions, average='macro'))
print("Total number of articles in test set it : ")
print(len(predictions))
