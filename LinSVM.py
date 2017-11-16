from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score,confusion_matrix
from sklearn.cross_validation import cross_val_score
from tfidf import * 
from start import *
from sklearn.cross_validation import train_test_split
from cross_validation import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer

def LinSVM():
    svm_headline = LinearSVC()
    svm_headline.fit(X_headline_tfidf_train, y_headline_train)
    y_svm_headline_pred = svm_headline.predict(X_headline_tfidf_test)


    print ("Linear SVM using News headlines for prediction\nF1 score,Accuracy,Precision and Recall : \n")
    print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_svm_headline_pred, average='macro')*100))
    print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_svm_headline_pred)*100))
    print ("Precision {:.4}%".format(precision_score(y_headline_test,y_svm_headline_pred) * 100))
    print ("Recall {:.4}%".format(recall_score(y_headline_test,y_svm_headline_pred) * 100))

    # print('Confusion Matrix: LinearSVM,Headlines\n')
    # cm = confusion_matrix(y_headline_test,y_svm_headline_pred,labels = ['FAKE','REAL'])
    # plot_confusion_matrix(cm,classes=['FAKE','REAL'])

    print ("Cross validation list and mean value(News headlines)\n")
    cros_val_list = cross_val_score(svm_headline, X_headline_tfidf,y,cv=5)
    print(cros_val_list)
    print(cros_val_list.mean())

    print ("Applying K-fold cross validation for Linear SVM(using news headlines)\n")
    xtrain,xtest,ytrain,ytest = train_test_split(X_headline_tfidf,y)
    cv = cross_validation(svm_headline, xtrain, ytrain , n_splits=5,init_chunk_size = 100, chunk_spacings = 10, average = "binary",title = "Linear SVM using News headlines")
    cv.validate_for_holdout_set(xtest, ytest)
    cv.plot_learning_curve()

    svm_body = LinearSVC()
    svm_body.fit(X_body_tfidf_train, y_body_train)
    y_svm_body_pred = svm_body.predict(X_body_tfidf_test)

    print ("Linear SVM using News bodies for prediction\nF1 score,Accuracy,Precision and Recall : \n")
    print ("F1 score {:.4}%".format( f1_score(y_headline_test, y_svm_body_pred, average='macro')*100))
    print ("Accuracy score {:.4}%".format(accuracy_score(y_headline_test, y_svm_body_pred)*100))
    print ("Precision {:.4}%".format(precision_score(y_headline_test,y_svm_body_pred) * 100))
    print ("Recall {:.4}%".format(recall_score(y_headline_test,y_svm_body_pred) * 100))

    # print('Confusion Matrix: Linear SVM,Bodies\n')
    # cm = confusion_matrix(y_body_test,y_body_pred,labels = ['FAKE','REAL'])
    # plot_confusion_matrix(cm,classes=['FAKE','REAL'])

    print ("Cross validation list and mean value(News bodies)\n")
    cros_val_list = cross_val_score(svm_body, X_body_tfidf,y,cv=5)
    print(cros_val_list)
    print(cros_val_list.mean())


    xtrain,xtest,ytrain,ytest = train_test_split(X_body_tfidf,y)

    print ("Applying K-fold cross validation for Linear SVM(using news bodies)\n")
    cv = cross_validation(svm_body, xtrain, ytrain , n_splits=5,init_chunk_size = 100, chunk_spacings = 10, average = "binary",title = "Linear SVM using News Bodies")
    cv.validate_for_holdout_set(xtest, ytest)
    cv.plot_learning_curve()