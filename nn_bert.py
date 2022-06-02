import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, AdamW, BertModel
import torch.nn as nn


class BERT_Arch(nn.Module):

    def __init__(self):
        super(BERT_Arch, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # freeze all the parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 3)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        print('GOING')
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)
        # x = self.dropout(x)
        x = self.softmax(x)
        return x


def train(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    for step, batch in enumerate(train_loader):

        # progress update after every 50 batches.
        if step % 2 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = criterion(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_loader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


def evaluate(model, train_loader, val_loader, epochs, optimizer, criterion, device):
    print("\nEvaluating...")
    model.eval()

    total_loss, total_accuracy = 0, 0

    total_preds = []

    for step, batch in enumerate(val_loader):

        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_loader)))

        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch

        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = criterion(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(val_loader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


def run(train_loader, val_loader, epochs):
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    model = BERT_Arch()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    # compute the class weights
    # class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)
    # print(class_wts)
    # [0.57743559 3.72848948]
    # convert class weights to tensor
    # weights = torch.tensor(class_wts, dtype=torch.float)
    # weights = weights.to(device)
    # cross_entropy = nn.NLLLoss(weight=weights)

    criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        train_loss, _ = train(model, train_loader, val_loader, epochs, optimizer, criterion, device)
        valid_loss, _ = evaluate(model, train_loader, val_loader, epochs, optimizer, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    # # load weights of best model
    # path = 'saved_weights.pt'
    # model.load_state_dict(torch.load(path))
    # # get predictions for test data
    # with torch.no_grad():
    #     preds = model(test_seq.to(device), test_mask.to(device))
    #     preds = preds.detach().cpu().numpy()
    # # model's performance
    # preds = np.argmax(preds, axis=1)
    # print(classification_report(test_y, preds))
    #
    # # confusion matrix
    # pd.crosstab(test_y, preds)
