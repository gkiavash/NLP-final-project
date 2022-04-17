from string import punctuation


def preprocess(text):
    text = text.lower()
    for punc in punctuation:
        text = text.replace(punc, ' ')

    return text
