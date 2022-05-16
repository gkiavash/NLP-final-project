import numpy as np
from collections import Counter

from torch import nn
from transformers import AutoTokenizer

import nn_model_2
from nn_model_2 import SentimentNet
import utils

tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")

# train_comments, train_labels = utils.load_data("face_masks_train_retrieved.tsv")
# test_comments, test_labels = utils.load_data("face_masks_test_retrieved.tsv")
train_comments, train_labels, test_comments, test_labels = utils.load_data_all("output")

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
# Statistics:
print(train_tokens)
print(test_tokens)
count_words = Counter(train_labels)
print(count_words.keys(), count_words.values())
count_words = Counter(test_labels)
print(count_words.keys(), count_words.values())


train_labels = utils.label_to_one_hot(train_labels)
test_labels = utils.label_to_one_hot(test_labels)

import torch
from torch.utils.data import TensorDataset, DataLoader

train_data = TensorDataset(torch.from_numpy(list(train_tokens.values())[0]), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(list(test_tokens.values())[0]), torch.from_numpy(test_labels))

batch_size = 200

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
    hidden_dim=3,
)
model.to(device)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

nn_model_2.train(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=5,
    optimizer=optimizer,
    criterion=criterion,
)
model.load_state_dict(torch.load('./state_dict.pt'))

test_losses = []
num_correct = 0

model.eval()
corrects = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # rounds the output to 0/1

    pred_class = torch.argmax(output, dim=1)
    labels_class = torch.argmax(labels, dim=1)
    corrects += torch.sum(pred_class == labels_class)


print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = corrects / len(test_labels)
print("Test accuracy: {:.3f}%".format(test_acc * 100))
print("corrects", corrects)

x_len = 0

for i in range(len(list(labels_class))):
    # print(labels_class[i], pred_class[i])
    if labels_class[i] == pred_class[i]:
        x_len += 1
print(x_len, len(labels_class))


