import csv
import os

import numpy as np
from string import punctuation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def preprocess(text):
    text = text.lower()
    for punc in punctuation:
        text = text.replace(punc, ' ')

    return text


def load_data(path_to_csv):
    comments = []
    labels = []

    with open(path_to_csv, encoding='utf-8') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line_index, line in enumerate(tsv_file):
            if line_index == 0:
                continue
            comments.append(preprocess(line[5]))
            labels.append(line[2])
    return comments, labels


def label_to_one_hot(labels):
    integer_encoded = LabelEncoder().fit_transform(np.array(labels))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = OneHotEncoder(sparse=False).fit_transform(integer_encoded)
    return onehot_encoded


def load_data_all(path_to_csv):
    filenames = os.listdir(path_to_csv)
    comments_train = []
    labels_train = []

    comments_test = []
    labels_test = []
    for filename in filenames:
        comments_, labels_ = load_data(os.path.join(path_to_csv, filename))
        if 'train' in filename:
            comments_train.extend(comments_)
            labels_train.extend(labels_)
        elif 'test' in filename:
            comments_test.extend(comments_)
            labels_test.extend(labels_)
    print(len(comments_train), len(labels_train), len(comments_test), len(labels_test))
    return comments_train, labels_train, comments_test, labels_test
