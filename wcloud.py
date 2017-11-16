from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import wordcloud
from start import *

stop_words = set(stopwords.words('english')) #list of stop words

file = ' '.join(dft.text.values)# Use this to read file content as a stream:
werd = list()
for r in file.split():
    #print(r)
    if not r in stop_words: #removing all stop words to build the word cloud
        werd.append(r)
        
realcloud = wordcloud.WordCloud(width = 1000, height = 500).generate(' '.join(werd))

plt.figure(figsize=(15,8))
plt.title('Word cloud for articles labeled as real')
plt.imshow(realcloud)
plt.axis("off")
plt.show()

file = ' '.join(dff.text.values)# Use this to read file content as a stream:
werd = list()
for r in file.split():
    #print(r)
    if not r in stop_words: #removing all stop words to build the word cloud
        werd.append(r)
        
fakecloud = wordcloud.WordCloud(width = 1000, height = 500).generate(' '.join(werd))

plt.figure(figsize=(15,8))
plt.title('Word cloud for articles labelled as fake')
plt.imshow(fakecloud)
plt.axis("off")
plt.show()