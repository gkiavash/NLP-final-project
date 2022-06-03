import numpy as np
from collections import Counter

from torch import nn
from transformers import AutoTokenizer

import nn_model_2
import nn_bert
import utils


INPUT_LENGTH = 128


tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")

train_comments, train_labels = utils.load_data("dataset_full/face_masks_train.csv")
test_comments, test_labels = utils.load_data("dataset_full/face_masks_test.csv")
val_comments, val_labels = utils.load_data("dataset_full/face_masks_val.csv")

# (
#     train_comments,
#     train_labels,
#     test_comments,
#     test_labels,
#     val_comments,
#     val_labels
# ) = utils.load_data_all("output")

train_tokens = tokenizer(
    train_comments,
    padding='max_length',
    max_length=INPUT_LENGTH,
    truncation=True,
    add_special_tokens=True,
    return_tensors="np"
)
test_tokens = tokenizer(
    test_comments,
    padding='max_length',
    max_length=INPUT_LENGTH,
    truncation=True,
    add_special_tokens=True,
    return_tensors="np"
)
val_tokens = tokenizer(
    val_comments,
    padding='max_length',
    max_length=INPUT_LENGTH,
    truncation=True,
    add_special_tokens=True,
    return_tensors="np"
)
# Statistics:
# print(train_tokens)
# print(test_tokens)
for lables in (train_labels, test_labels, val_labels):
    count_words = Counter(lables)
    print(count_words.keys(), count_words.values())


train_labels = utils.label_to_one_hot(train_labels)
test_labels = utils.label_to_one_hot(test_labels)
val_labels = utils.label_to_one_hot(val_labels)

import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(
    torch.from_numpy(list(train_tokens.values())[0]),  # data
    torch.from_numpy(list(train_tokens.values())[1]),  # mask
    torch.from_numpy(train_labels)                     # label
)
test_data = TensorDataset(
    torch.from_numpy(list(test_tokens.values())[0]),
    torch.from_numpy(list(test_tokens.values())[1]),
    torch.from_numpy(test_labels)
)
val_data = TensorDataset(
    torch.from_numpy(list(val_tokens.values())[0]),
    torch.from_numpy(list(val_tokens.values())[1]),
    torch.from_numpy(val_labels)
)

batch_size = 200

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)

dataiter = iter(train_loader)
sample_x, sample_mask, sample_y = dataiter.next()
print(sample_x.shape, sample_y.shape)

nn_bert.run(train_loader, val_loader, 10)
#
# nn_model_2.run(train_loader, val_loader, test_loader, epochs=20, INPUT_LENGTH=INPUT_LENGTH)
