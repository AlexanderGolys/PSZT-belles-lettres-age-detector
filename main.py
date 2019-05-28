import pandas as pd
import numpy as np
import os
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from json import loads
import urllib


def clean_text(text):
    res = ''
    for char in text:
        if char not in string.punctuation and char != '0' and char != '1' and char != '2' and char != '/n':
            res += char.lower()
    return res


def text_process(text):
    return [word for word in text.split()]



text = [line.rstrip() + ' ' for line in open('./data_renesans.csv')]
data = [text[0]]

for i in range(len(text)):
    if text[i] != '' and (text[i][0] == '0' or text[i][0] == '1' or text[i][0] == '2'
    or text[i][0] == '3' or text[i][0] == '4' or text[i][0] == '5' or text[i][0] == '6'):
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
print(data[data['a'] == '3'])
text_train, text_test, epoch_train, epoch_test = train_test_split(data['b'], data['a'], test_size=0.2)

pipeline = Pipeline([
   ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

pipeline.fit(text_train, epoch_train)

pred = pipeline.predict(text_test)

print('renesans: ', data['a'][data['a'] == '0'].count())
print('barok: ', data['a'][data['a'] == '1'].count())

print(classification_report(pred, epoch_test))


# oswiecenie = loads(urllib.request.urlopen("https://wolnelektury.pl/katalog/epoka/oswiecenie/").read().decode('utf-8'))
# print(oswiecenie)
