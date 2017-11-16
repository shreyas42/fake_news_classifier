from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score,confusion_matrix
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cross_validation import train_test_split
from cross_validation import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from start import *



##################TF-IDF Vectorizer to generate the feature vectors for the list of documents################
tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,2),max_df= 0.85, min_df= 0.01)

#feature vectors for the news bodies
X_body_tfidf = tfidf.fit_transform(X_body_text)
#feature vectors for the news titles
X_headline_tfidf = tfidf.fit_transform(X_headline_text)

#splitting the feature space into training and testing set
X_headline_tfidf_train, X_headline_tfidf_test, y_headline_train, y_headline_test = train_test_split(X_headline_tfidf,y, test_size = 0.2, random_state=1234)
X_body_tfidf_train, X_body_tfidf_test, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)

