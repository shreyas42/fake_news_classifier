
from LogRegression import LogRegression
from MultiNB import MultiNB
from LinSVM import LinSVM
from start import *
from RNN import LSTM_fakenews
from RandForest import RandForest

if __name__ == '__main__':    
    choice = int(input('Enter 1 for Logistic Regression\n2 for Multinomial Naive Bayes\n3 for Linear SVM\n4 for LSTM\n5 for Random Forest'))
    if(choice == 1):
        LogRegression()
    elif(choice == 2):
        MultiNB()
    elif(choice == 3):
        LinSVM()
    elif(choice == 4):
        LSTM_fakenews()
    elif(choice == 5):
        RandForest()
