import numpy as np
import pandas as pd
import os
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def answer(ans):
    if ans[0] == '0':
        print("renesans")
    elif ans[0] == '1':
        print("barok")
    elif ans[0] == '2':
        print("oswiecenie")
    elif ans[0] == '3':
        print("romantyzm")


def add_data(texts, char_nb, data):
    for i in range(len(texts)):
        if (texts[i] != '' and texts[i][0] == char_nb) or (len(texts[i]) > 1 and texts[i][1] == char_nb and texts[i][0] == ' '):
            data.append(texts[i])
            count = 0
        elif count < 21:
            data[-1] += texts[i]
            count += 1


def clean_text(text):
    res = ''
    for char in text:
        if char not in string.punctuation and char != '/n' and char != '*' and char != '-' and char != '«' and char != '»':
            res += char.lower()
    if len(text) > 0 and text[0] == ' ':
        return res[1:]
    else:
        return res


def text_process(text):
    return [word for word in text.split()]


def main():
    text = [line.rstrip() + ' ' for line in open('./data_renesans.csv')]
    text2 = [line.rstrip() + ' ' for line in open('./data_Oswiecenie_300.txt')]
    text3 = [line.rstrip() + ' ' for line in open('./data_Romantyzm_300.txt')]  #read data from files
    data = [text[0]]

    count = 0
    for i in range(len(text)):   #divide data to singular texts
        if text[i] != '' and (text[i][0] == '0' or text[i][0] == '1'):
            data.append(text[i])
            count = 0
        elif count < 21:
            data[-1] += text[i]
            count += 1

    add_data(text2, '2', data)  # add data from last 2 epochs
    data = data[:-80]  # rebalances number of samples
    add_data(text3, '3', data)
    data = data[:-60]

    clean_data = [clean_text(line) for line in data]  # cleaning texts
    better_data = []
    for text in clean_data:  # divide data into epoch number and text
        better_data.append([text[0], text[2:]])
    better_data = better_data[1:]  # delete repetition

    data = pd.DataFrame(np.array(better_data), columns=['a', 'b'])  # creating pandas dataframe from data
    text_train, text_test, epoch_train, epoch_test = train_test_split(data['b'], data['a'], test_size=0.3)  # split data

    print(data.describe(), data.a.unique())
    print('renesans: ', data['a'][data['a'] == '0'].count())
    print('barok: ', data['a'][data['a'] == '1'].count())
    print('oswiecenie: ', data['a'][data['a'] == '2'].count())
    print('romantyzm: ', data['a'][data['a'] == '3'].count())


    pipeline = Pipeline([  # create classifier
       ('bow', CountVectorizer(analyzer=text_process)),  # create bucket of words
        ('tfidf', TfidfTransformer()),  # create tf-idf from bow
        ('classifier', MultinomialNB()),  # create Naive Bayes
    ])

    pipeline.fit(text_train, epoch_train)  # feed classifier with data

    pred = pipeline.predict(text_test)  # check accuracy on test data
    print(classification_report(pred, epoch_test))  # print report from checking
    text = ''
    for line in open('./text.txt'):  # read text to be classify
        text += line
    text = clean_text(text)  # clean text to be classify
    print(text)
    answer(pipeline.predict([text]))  # print answer


main()

