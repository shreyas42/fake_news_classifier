import pandas as pd
from sklearn import preprocessing
df = pd.read_csv('data/updated.csv')
df = df.dropna()
#depends on available RAM
df = df.sample(10000)
dft  = df.loc[df['label'] == 'REAL'] 
dff = df.loc[df['label'] == 'FAKE']


X_body_text = df.text.values #list of news body documents
X_headline_text = df.title.values #list of news title documents
y = df.label.values #list of labels for each document
# 1. INSTANTIATE
enc = preprocessing.LabelEncoder()

# 2. FIT
enc.fit(y)

# 3. Transform
y = enc.transform(y)

