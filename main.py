from collections import Counter
from cross_validation import cross_validation
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import io
import itertools
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import wordcloud

df = pd.read_csv('data/updated.csv')
df = df.dropna()
#depends on available RAM
df = df.sample(5000)
dft  = df.loc[df['label'] == 'REAL'] 


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


##################WORD CLOUD###########################
stop_words = set(stopwords.words('english')) #list of stop words

file = ' '.join(dft.text.values)# Use this to read file content as a stream:
werd = list()
for r in file.split():
    #print(r)
    if not r in stop_words: #removing all stop words to build the word cloud
        werd.append(r)
        
wordcloud = wordcloud.WordCloud(width = 1000, height = 500).generate(' '.join(werd))

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

###########Preparing the target and predictors for modeling####################

X_body_text = df.text.values #list of news body documents
X_headline_text = df.title.values #list of news title documents
y = df.label.values #list of labels for each document

###########encoding labels###########
# 1. INSTANTIATE
enc = preprocessing.LabelEncoder()

# 2. FIT
enc.fit(y)

# 3. Transform
y = enc.transform(y)


##################TF-IDF Vectorizer to generate the feature vectors for the list of documents################
tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2),max_df= 0.85, min_df= 0.01)

#feature vectors for the news bodies
X_body_tfidf = tfidf.fit_transform(X_body_text)
#feature vectors for the news titles
X_headline_tfidf = tfidf.fit_transform (X_headline_text)

#splitting the feature space into training and testing set
X_headline_tfidf_train, X_headline_tfidf_test, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
X_body_tfidf_train, X_body_tfidf_test, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)


'''
############# LOGISTIC REGRESSION ##################################

print ("Logistic Regression")
lr_headline = LogisticRegression(penalty='l1')
# train model for the headlines
lr_headline.fit(X_headline_tfidf_train, y_headline_train)

# get predictions based on the titles of articles
y_headline_pred = lr_headline.predict(X_headline_tfidf_test)

# print metrics
print ("Logistic Regression using News headlines for prediction\nF1 score,Accuracy,Precision and Recall : \n")
print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_headline_pred, average='macro')*100))
print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_headline_pred)*100))
print ("Precision {:.4}%".format(precision_score(y_headline_test,y_headline_pred) * 100))
print ("Recall {:.4}%".format(recall_score(y_headline_test,y_headline_pred) * 100))

print('Confusion Matrix: Logistic Regression,Headlines\n')
cm = confusion_matrix(y_headline_test,y_headline_pred,labels = ['FAKE','REAL'])
plot_confusion_matrix(cm,classes=['FAKE','REAL'])

print ("Cross validation list and mean value(News headlines)\n")
cros_val_list = cross_val_score(lr_headline, X_headline_tfidf,y,cv=7)
print(cros_val_list)
print(cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)

#cross validation to ensure correctness of sampling

print ("Applying K-fold cross validation for Logistic Regression(using news headlines)\n")
cv = cross_validation(lr_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 100, chunk_spacings = 10, average = "binary",title = "Logistic Regression on news headlines")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

lr_body = LogisticRegression(penalty='l1')
lr_body.fit(X_body_tfidf_train, y_body_train)
y_lr_body_pred = lr_body.predict(X_body_tfidf_test)

# print metrics
print ("Logistic Regression using News bodies for prediction\nF1 score,Accuracy,Precision and Recall : \n")
print ("F1 score {:.4}%".format( f1_score(y_body_test, y_lr_body_pred, average='macro')*100))
print ("Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_lr_body_pred)*100))
print ("Precision {:.4}%".format(precision_score(y_body_test,y_lr_body_pred) * 100))
print ("Recall {:.4}%".format(recall_score(y_body_test,y_lr_body_pred) * 100))

print('Confusion Matrix: Logistic Regression,Bodies\n')
cm = confusion_matrix(y_body_test,y_lr_body_pred,labels = ['FAKE','REAL'])
plot_confusion_matrix(cm,classes=['FAKE','REAL'])

print ("Cross validation list and mean value(News bodies)\n")
cros_val_list = cross_val_score(lr_body, X_body_tfidf,y,cv=7)
print(cros_val_list)
print(cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

print ("Applying K-fold cross validation for Logistic Regression(using news bodies)\n")
cv = cross_validation(lr_body, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary",title = "Logistic Regression on news bodies")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

################ Random Forest ######################################
print('Random forest')
rcf_headline = RandomForestClassifier(n_estimators=100,n_jobs=3)
rcf_headline.fit(X_headline_tfidf_train, y_headline_train)
y_rc_headline_pred = rcf_headline.predict(X_headline_tfidf_test)


print ("Random Forest using News headlines for prediction\nF1 score,Accuracy,Precision and Recall : \n")
print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_rc_headline_pred, average='macro')*100))
print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_rc_headline_pred)*100))
print ("Precision {:.4}%".format(precision_score(y_headline_test,y_rc_headline_pred) * 100))
print ("Recall {:.4}%".format(recall_score(y_headline_test,y_rc_headline_pred) * 100))

print('Confusion Matrix: Random Forest,Headlines\n')
cm = confusion_matrix(y_headline_test,y_rc_headline_pred,labels = ['FAKE','REAL'])
plot_confusion_matrix(cm,classes=['FAKE','REAL'])

print ("Cross validation list and mean value(News headlines)\n")
cros_val_list = cross_val_score(rcf_headline, X_headline_tfidf,y,cv=5)
print(cros_val_list)
print(cros_val_list.mean())

print ("Applying K-fold cross validation for Random Forest(using news headlines)\n")
xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)
cv = cross_validation(rcf_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary",title = "Random forest using News headlines")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

rcf_body = RandomForestClassifier(n_estimators=100,n_jobs=3)
rcf_body.fit(X_body_tfidf_train, y_body_train)
y_rc_body_pred = rcf_body.predict(X_body_tfidf_test)

print ("Random Forest using News bodies for prediction\nF1 score,Accuracy,Precision and Recall : \n")
print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_rc_body_pred, average='macro')*100))
print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_rc_body_pred)*100))
print ("Precision {:.4}%".format(precision_score(y_headline_test,y_rc_body_pred) * 100))
print ("Recall {:.4}%".format(recall_score(y_headline_test,y_rc_body_pred) * 100))

print('Confusion Matrix: Random Forest,Bodies\n')
cm = confusion_matrix(y_body_test,y_rc_body_pred,labels = ['FAKE','REAL'])
plot_confusion_matrix(cm,classes=['FAKE','REAL'])

print ("Cross validation list and mean value(News bodies)\n")
cros_val_list = cross_val_score(rcf_body, X_body_tfidf,y,cv=5)
print(cros_val_list)
print(cros_val_list.mean())


xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

print ("Applying K-fold cross validation for Random Forest(using news bodies)\n")
cv = cross_validation(rcf_body, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary",title = "Random forest using News Bodies")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

#############Linear SVM ##################################

print('Linear SVM')
svm_headline = LinearSVC()
svm_headline.fit(X_headline_tfidf_train, y_headline_train)
y_svm_headline_pred = svm_headline.predict(X_headline_tfidf_test)


print ("Linear SVM using News headlines for prediction\nF1 score,Accuracy,Precision and Recall : \n")
print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_svm_headline_pred, average='macro')*100))
print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_svm_headline_pred)*100))
print ("Precision {:.4}%".format(precision_score(y_headline_test,y_svm_headline_pred) * 100))
print ("Recall {:.4}%".format(recall_score(y_headline_test,y_svm_headline_pred) * 100))

print('Confusion Matrix: LinearSVM,Headlines\n')
cm = confusion_matrix(y_headline_test,y_svm_headline_pred,labels = ['FAKE','REAL'])
plot_confusion_matrix(cm,classes=['FAKE','REAL'])

print ("Cross validation list and mean value(News headlines)\n")
cros_val_list = cross_val_score(svm_headline, X_headline_tfidf,y,cv=5)
print(cros_val_list)
print(cros_val_list.mean())

print ("Applying K-fold cross validation for Linear SVM(using news headlines)\n")
xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)
cv = cross_validation(svm_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 100, chunk_spacings = 10, average = "binary",title = "Linear SVM using News headlines")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

svm_body = RandomForestClassifier(n_estimators=100,n_jobs=3)
svm_body.fit(X_body_tfidf_train, y_body_train)
y_svm_body_pred = svm_body.predict(X_body_tfidf_test)

print ("Linear SVM using News bodies for prediction\nF1 score,Accuracy,Precision and Recall : \n")
print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_svm_body_pred, average='macro')*100))
print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_svm_body_pred)*100))
print ("Precision {:.4}%".format(precision_score(y_headline_test,y_svm_body_pred) * 100))
print ("Recall {:.4}%"....