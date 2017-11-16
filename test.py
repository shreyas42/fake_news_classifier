
import pandas as pd
from collections import Counter
import re
import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from cross_validation import cross_validation
import wordcloud

df = pd.read_csv('DefFinal.csv')
df = df.dropna()
df = df.sample(5000)
dft  = df.loc[df['label'] == 'REAL'] 
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#word_tokenize accepts a string as an input, not a file.
stop_words = set(stopwords.words('english'))

file = ' '.join(dft.text.values)# Use this to read file content as a stream:
print(type(file))
werd = list()
for r in file.split():
    #print(r)
    if not r in stop_words:
        werd.append(r)
        
wordcloud = wordcloud.WordCloud(width = 1000, height = 500).generate(' '.join(werd))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# Preparing the target and predictors for modeling

X_body_text = df.text.values
X_headline_text = df.title.values
y = df.label.values

# 1. INSTANTIATE
enc = preprocessing.LabelEncoder()

# 2. FIT
enc.fit(y)

# 3. Transform
y = enc.transform(y)

tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2),max_df= 0.85, min_df= 0.01)

X_body_tfidf = tfidf.fit_transform(X_body_text)
X_headline_tfidf = tfidf.fit_transform (X_headline_text)

X_headline_tfidf_train, X_headline_tfidf_test, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
X_body_tfidf_train, X_body_tfidf_test, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)


'''s
############# LOGISTIC REGRESSION ##################################

lr_headline = LogisticRegression(penalty='l1')
# train model
lr_headline.fit(X_headline_tfidf_train, y_headline_train)

# get predictions for article section
y_headline_pred = lr_headline.predict(X_headline_tfidf_test)

# print metrics
print ("Logistig Regression F1 and Accuracy Scores : \n")
print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_headline_pred, average='macro')*100))
print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_headline_pred)*100))

cros_val_list = cross_val_score(lr_headline, X_headline_tfidf,y,cv=7)
print(cros_val_list)
print(cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)

cv = cross_validation(lr_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 100, chunk_spacings = 10, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

lr_body = LogisticRegression(penalty='l1')
lr_body.fit(X_body_tfidf_train, y_body_train)
y_lr_body_pred = lr_body.predict(X_body_tfidf_test)
# print metrics
print ("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_lr_body_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_lr_body_pred)*100) )

xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

cv = cross_validation(lr_body, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()


################ Random Forest ######################################

rcf_headline = RandomForestClassifier(n_estimators=100,n_jobs=3)
rcf_headline.fit(X_headline_tfidf_train, y_headline_train)
y_rc_headline_pred = rcf_headline.predict(X_headline_tfidf_test)


# print metrics
print ("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_rc_headline_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_rc_headline_pred)*100) )

cros_val_list = cross_val_score(rcf_headline, X_headline_tfidf,y,cv=5)
print(cros_val_list)
print(cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)
cv = cross_validation(rcf_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

rcf_body = RandomForestClassifier(n_estimators=100,n_jobs=3)
rcf_body.fit(X_body_tfidf_train, y_body_train)
y_rc_body_pred = rcf_body.predict(X_body_tfidf_test)
# print metrics
print ("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_rc_body_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_rc_body_pred)*100) )

xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

cv = cross_validation(rcf_body, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()



'''
############# ##################################

lr_body = LinearSVC()
# train model
lr_body.fit(X_body_tfidf_train, y_body_train)

# get predictions for article section
y_body_pred= lr_body.predict(X_body_tfidf_test)
print(type(y_body_pred))
print(type(y_body_test))
print(y_body_test)
print(y_body_pred)
#y_body_test
pred_label = list()
for i in y_body_pred:
    if(i == 0):
        pred_label.append("REAL")
    else:
        pred_label.append("FAKE")
pred_l = np.array(pred_label)
y_body_test["pred"] = pred_label 
# print metrics
print ("SVM F1 and Accuracy Scores : \n")
print ("F1 score {:.4}%".format( f1_score(y_body_test, y_body_pred, average='macro')*100))
print ("Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_body_pred)*100))

cros_val_list = cross_val_score(lr_body, X_body_tfidf,y,cv=7)
print(cros_val_list)
print(cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

cv = cross_validation(lr_body, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

svm_headline = LinearSVC(n_estimators=100,n_jobs=3)
svm_headline.fit(X_headline_tfidf_train, y_headline_train)
y_svm_headline_pred = svm_headline.predict(X_headline_tfidf_test)


# print metrics
print ("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_svm_headline_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_svm_headline_pred)*100) )

cros_val_list = cross_val_score(svm_headline, X_headline_tfidf,y,cv=5)
print(cros_val_list)
print(cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)
cv = cross_validation(svm_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

############# Multinomial NB ##################################
mnb_headline = RandomForestClassifier(n_estimators=100,n_jobs=3)
mnb_headline.fit(X_headline_tfidf_train, y_headline_train)
y_mnb_headline_pred = mnb_headline.predict(X_headline_tfidf_test)


# print metrics
print ("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_headline_test, y_mnb_headline_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_mnb_headline_pred)*100) )

cros_val_list = cross_val_score(mnb_headline, X_headline_tfidf,y,cv=5)
print(cros_val_list)
print(cros_val_list.mean())

xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)
cv = cross_validation(mnb_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()

mnb_body = RandomForestClassifier(n_estimators=100,n_jobs=3)
mnb_body.fit(X_body_tfidf_train, y_body_train)
y_mnb_body_pred = mnb_body.predict(X_body_tfidf_test)
# print metrics
print ("Random Forest F1 and Accuracy Scores : \n")
print ( "F1 score {:.4}%".format( f1_score(y_body_test, y_mnb_body_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%".format(accuracy_score(y_body_test, y_mnb_body_pred)*100) )

xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

cv = cross_validation(mnb_body, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary")
cv.validate_for_holdout_set(xtest, ytest)
cv.plot_learning_curve()
############# Multinomial NB ##################################




########### LSTM ##################################################
from keras.preprocessing.text.Tokenizer import Tokenizer

xlist = list(X_train)
#print(xlist)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(xlist)
print(len(tokenizer.word_index))
sequences = tokenizer.texts_to_sequences(xlist)
#print(sequences)
l = len(max(sequences,key = lambda  x : len(x)))
print(l)
padded_sequences = pad_sequences(sequences, maxlen = 1000) #padded_sequencies is the tokenized and padded data
#padded_sequences

model = Sequential()
model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=1000)) #maxlen of tokenizerwordindex
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) #128 depends on no of words in a row 
model.add(Dense(2, activation='sigmoid')) #2 because of one hot enc
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.utils import to_categorical

y_train_new = []
y_test_new = []
for x in y_train:
    if x == 'sarc':
        y_train_new.append(1)
    else:
        y_train_new.append(0)
for x in y_test:
    if x == 'sarc':
        y_test_new.append(1)
    else:
        y_test_new.append(0)
        
                
#print(y_train_new)
y_train_new = to_categorical(y_train_new, num_classes = 2)
y_test_new = to_categorical(y_test_new,  num_classes = 2) 
#print(y_train_new)

model.fit(padded_sequences, y_train_new, validation_split=0.2, epochs=3)

xlist_test = list(X_test)
#print(xlist)


#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(xlist_test)
print(len(tokenizer.word_index))
sequences = tokenizer.texts_to_sequences(xlist_test)
#print(sequences)
l_test = len(max(sequences,key = lambda  x : len(x)))
print(l_test)
padded_sequences_test = pad_sequences(sequences, maxlen = 1000) #padded_sequencies is the tokenized and padded data
#padded_sequences

scores = model.evaluate(padded_sequences_test,y_test_new,verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#y_pred = model.predict()














