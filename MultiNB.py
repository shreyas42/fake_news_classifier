from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score,confusion_matrix
from sklearn.cross_validation import cross_val_score
from tfidf import *
from start import * 
from sklearn.cross_validation import train_test_split
from cross_validation import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer

############# Multinomial NB ##################################
def MultiNB():
    mnb_headline = MultinomialNB()
    mnb_headline.fit(X_headline_tfidf_train, y_headline_train)
    y_mnb_headline_pred = mnb_headline.predict(X_headline_tfidf_test)


    print ("Multinomial Naive Bayes using News headlines for prediction\nF1 score,Accuracy,Precision and Recall : \n")
    print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_mnb_headline_pred, average='macro')*100))
    print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_mnb_headline_pred)*100))
    print ("Precision {:.4}%".format(precision_score(y_headline_test,y_mnb_headline_pred) * 100))
    print ("Recall {:.4}%".format(recall_score(y_headline_test,y_mnb_headline_pred) * 100))

    # print('Confusion Matrix: Multinomial Naive Bayes,Headlines\n')
    # cm = confusion_matrix(y_headline_test,y_mnb_headline_pred,labels = ['FAKE','REAL'])
    # plot_confusion_matrix(cm,classes=['FAKE','REAL'])

    print ("Cross validation list and mean value(News headlines)\n")
    cros_val_list = cross_val_score(mnb_headline, X_headline_tfidf,y,cv=5)
    print(cros_val_list)
    print(cros_val_list.mean())

    print ("Applying K-fold cross validation for Multinomial Naive Bayes(using news headlines)\n")
    xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)
    cv = cross_validation(mnb_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 100, chunk_spacings = 10, average = "binary",title = "Multinomial Naive Bayes using News headlines")
    cv.validate_for_holdout_set(xtest, ytest)
    cv.plot_learning_curve()

    mnb_body = MultinomialNB()
    mnb_body.fit(X_body_tfidf_train, y_body_train)
    y_mnb_body_pred = mnb_body.predict(X_body_tfidf_test)

    print ("Multinomial Naive Bayes using News bodies for prediction\nF1 score,Accuracy,Precision and Recall : \n")
    print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_mnb_body_pred, average='macro')*100))
    print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_mnb_body_pred)*100))
    print ("Precision {:.4}%".format(precision_score(y_headline_test,y_mnb_body_pred) * 100))
    print ("Recall {:.4}%".format(recall_score(y_headline_test,y_mnb_body_pred) * 100))

    # print('Confusion Matrix: Multinomial Naive Bayes,Bodies\n')
    # cm = confusion_matrix(y_body_test,y_mnb_body_pred,labels = ['FAKE','REAL'])
    # plot_confusion_matrix(cm,classes=['FAKE','REAL'])

    print ("Cross validation list and mean value(News bodies)\n")
    cros_val_list = cross_val_score(mnb_body, X_body_tfidf,y,cv=5)
    print(cros_val_list)
    print(cros_val_list.mean())


    xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

    print ("Applying K-fold cross validation for Multinomial Naive Bayes(using news bodies)\n")
    cv = cross_validation(mnb_body, xtrain, ytrain , n_splits=5,init_chunk_size = 100, chunk_spacings = 10, average = "binary",title = "Multinomial Naive Bayes using News Bodies")
    cv.validate_for_holdout_set(xtest, ytest)
    cv.plot_learning_curve()
    ############# Multinomial NB ##################################
