from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer 	# Using inverse document frequency to filter the noise
from sklearn.pipeline import Pipeline
from sklearn import metrics
from global_variables import stop, cats
import time

train_dataset = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
# remove=('headers', 'footers', 'quotes'), 
# Using pipeline to fit data to model and then convert it to tf-idf counts
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(stop_words=stop, ngram_range=(1,  2))),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier',     SGDClassifier(alpha=1e-3, random_state=42)    )
])

start_time = time.process_time()

# SGDClassifier(alpha=1e-3, random_state=42)
pipeline.fit(train_dataset.data, train_dataset.target)
print("Training Time : "+str(time.process_time()-start_time))

test_dataset = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=cats)

start_time = time.process_time()
predictions = pipeline.predict(test_dataset.data)
actual_answers = test_dataset.target
print("Prediction Time : "+str(time.process_time()-start_time))

print("The Accuracy is : "+str(metrics.accuracy_score(actual_answers, predictions, normalize=True)*100))
print("The F-Score is : "+str(metrics.f1_score(actual_answers, predictions, average='macro')))
print("Total number of articles in the training set is : "+str(len(train_dataset.target)))
print("Total number of articles in the test set is : "+str(len(predictions)))