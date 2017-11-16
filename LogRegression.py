from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score,confusion_matrix
from sklearn.cross_validation import cross_val_score
from tfidf import * 
from sklearn.cross_validation import train_test_split
from cross_validation import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer


############# LOGISTIC REGRESSION ##################################
def LogRegression():
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

    # print('Confusion Matrix: Logistic Regression,Headlines\n')
    # cm = confusion_matrix(y_headline_test,y_headline_pred,labels = ['FAKE','REAL'])
    # plot_confusion_matrix(cm,classes=['FAKE','REAL'])

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

    # print('Confusion Matrix: Logistic Regression,Bodies\n')
    # cm = confusion_matrix(y_body_test,y_lr_body_pred,labels = ['FAKE','REAL'])
    # plot_confusion_matrix(cm,classes=['FAKE','REAL'])

    print ("Cross validation list and mean value(News bodies)\n")
    cros_val_list = cross_val_score(lr_body, X_body_tfidf,y,cv=7)
    print(cros_val_list)
    print(cros_val_list.mean())

    xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

    print ("Applying K-fold cross validation for Logistic Regression(using news bodies)\n")
    cv = cross_validation(lr_body, xtrain, ytrain , n_splits=5,init_chunk_size = 1000, chunk_spacings = 10, average = "binary",title = "Logistic Regression on news bodies")
    cv.validate_for_holdout_set(xtest, ytest)
    cv.plot_learning_curve()