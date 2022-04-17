import csv
from string import punctuation


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


