import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold,train_test_split


inputFile = "finalData.csv"

csvfile = pd.read_csv(inputFile)

docs = []

for text in csvfile['text']:
    docs.append(text)

#tf-idf vectorizer

X_train,X_test = train_test_split(csvfile,test_size = 0.1,random_state = 42)

vectorizer = TfidfVectorizer(min_df = 5,max_df = 0.8,sublinear_tf = True,use_idf = True,stop_words = 'english')

print X_train
print X_test