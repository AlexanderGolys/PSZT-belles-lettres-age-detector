import pandas as pd
import numpy as np
import os
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB



def clean_text(text):
    res = ''
    for char in text:
        if char not in string.punctuation:
            res += char
        else:
            res += ' '
    return res


def text_process(text):
    return [word for word in text.split()]


text = [line.rstrip() for line in open('./data.csv')]
data = [text[0]]
for i in range(len(text)):
    if text[i] != '' and text[i][0] == '0':
        data.append(text[i])
    else:
        data[-1] += text[i]
better_data = []
for text in data:
    better_data.append([text[0], text[2:]])
better_data = better_data[1:]

for i in range(len(better_data)):
    better_data[i][1] = clean_text(better_data[i][1])

data = pd.DataFrame(np.array(better_data), columns=['a', 'b'])

counter = CountVectorizer(analyzer=text_process).fit(data['b'])
text_bow = counter.transform(data['b'])
print(text_bow.shape)
tfidf_transformer = TfidfTransformer().fit(text_bow)
text_tfidf = tfidf_transformer.transform(text_bow)
print(text_tfidf.shape)
classifier = MultinomialNB().fit(text_tfidf, message['a'])


