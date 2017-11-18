import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
########### LSTM ##################################################
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def LSTM_fakenews():
    df = pd.read_csv('data/updated.csv', encoding="utf-8")
    df = df.dropna()
    #
    # uncomment this line if your computer does not have suffienct RAM(at least 16GB)
    # df = df.sample(10000)

    # Preparing the target and predictors for modeling

    X_body_text = df.text.values
    X_headline_text = df.title.values
    y = df.label.values

    X_headline_train, X_headline_test, y_headline_train, y_headline_test = train_test_split(X_headline_text, y, test_size=0.2, random_state=0)

    X_body_train, X_body_test, y_body_train, y_body_test = train_test_split(X_body_text, y, test_size=0.2, random_state=0)

    X_headline_list = list(X_headline_train)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_headline_list)
    print(len(tokenizer.word_index))
    sequences = tokenizer.texts_to_sequences(X_headline_list)
    l = len(max(sequences,key = lambda  x : len(x)))
    padded_headline_sequences = pad_sequences(sequences, maxlen = 1000) #padded_sequencies is the tokenized and padded data
    #padded_sequences

    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=1000)) #maxlen of tokenizerwordindex
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) #128 depends on no of words in a row 
    model.add(Dense(2, activation='sigmoid')) #2 because of one hot enc
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    from keras.utils import to_categorical

    y_train_LSTM = []
    y_test_LSTM = []
    for x in y_headline_train:
        if x == 'REAL':
            y_train_LSTM.append(1)
        else:
            y_train_LSTM.append(0)
    for x in y_headline_test:
        if x == 'REAL':
            y_test_LSTM.append(1)
        else:
            y_test_LSTM.append(0)
            
                    
    y_train_LSTM = to_categorical(y_train_LSTM, num_classes = 2)
    y_test_LSTM = to_categorical(y_test_LSTM,  num_classes = 2) 

    model.fit(padded_headline_sequences, y_train_LSTM, validation_split=0.2, epochs=3)

    test = list(X_headline_test)
    print(len(tokenizer.word_index))
    sequences = tokenizer.texts_to_sequences(test)
    #Sequenized and padded data
    pad_test = pad_sequences(sequences, maxlen = 1000) 

    scores = model.evaluate(pad_test,y_test_LSTM,verbose=0)
    print("Accuracy using only headlines: %.2f%%" % (scores[1]*100))



    from keras.utils import to_categorical

    y_train_LSTM = []
    y_test_LSTM = []
    for x in y_body_train:
        if x == 'REAL':
            y_train_LSTM.append(1)
        else:
            y_train_LSTM.append(0)
    for x in y_body_test:
        if x == 'REAL':
            y_test_LSTM.append(1)
        else:
            y_test_LSTM.append(0)
        
                
    y_train_LSTM = to_categorical(y_train_LSTM, num_classes = 2)
    y_test_LSTM = to_categorical(y_test_LSTM,  num_classes = 2)


    X_body_list = list(X_body_train)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_body_list)
    print(len(tokenizer.word_index))
    sequences = tokenizer.texts_to_sequences(X_body_list)
    l = len(max(sequences,key = lambda  x : len(x)))
    padded_body_sequences = pad_sequences(sequences, maxlen = 1000) #padded_sequencies is the tokenized and padded data
    #padded_sequences

    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 128, input_length=1000)) #maxlen of tokenizerwordindex
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2)) #128 depends on no of words in a row 
    model.add(Dense(2, activation='sigmoid')) #2 because of one hot enc
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(padded_body_sequences, y_train_LSTM, validation_split=0.2, epochs=3)

    test = list(X_body_test)
    print(len(tokenizer.word_index))
    sequences = tokenizer.texts_to_sequences(test)
    #Sequenized and padded data
    pad_test = pad_sequences(sequences, maxlen = 1000) 

    scores = model.evaluate(pad_test,y_test_LSTM,verbose=0)
    print("Accuracy using only body: %.2f%%" % (scores[1]*100))


