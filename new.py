import csv
import numpy as np
from collections import Counter

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import nn_model_2
from nn_model_2 import SentimentNet
from utils import preprocess, load_data

tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")

train_comments, train_labels = load_data("face_masks_train_retrieved.tsv")
test_comments, test_labels = load_data("face_masks_test_retrieved.tsv")

train_tokens = tokenizer(
    train_comments,
    padding='max_length',
    max_length=128,
    truncation=True,
    add_special_tokens=True,
    return_tensors="np"
)
test_tokens = tokenizer(
    test_comments,
    padding='max_length',
    max_length=128,
    truncation=True,
    add_special_tokens=True,
    return_tensors="np"
)
print(train_tokens)
print(test_tokens)

count_words = Counter(train_labels)
print(count_words.keys())
print(count_words.values())


def label_to_one_hot(labels):
    integer_encoded = LabelEncoder().fit_transform(np.array(labels))
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = OneHotEncoder(sparse=False).fit_transform(integer_encoded)
    return onehot_encoded


train_labels = label_to_one_hot(train_labels)
test_labels = label_to_one_hot(test_labels)

import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(list(train_tokens.values())[0]), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(list(test_tokens.values())[0]), torch.from_numpy(test_labels))

batch_size = 64

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

is_cuda = torch.cuda.is_available()
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print(sample_x.shape, sample_y.shape)

device = nn_model_2.device

# vocab_size = len(word2idx) + 1`

model = SentimentNet(
    input_size=128,
    output_size=3,
    hidden_dim=64,
)
model.to(device)
print(model)

lr = 0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

nn_model_2.train(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    batch_size=batch_size,
    optimizer=optimizer,
    criterion=criterion,
)
model.load_state_dict(torch.load('./state_dict.pt'))

test_losses = []
num_correct = 0

model.eval()
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc * 100))
