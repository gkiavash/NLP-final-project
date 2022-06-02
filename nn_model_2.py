import numpy as np
import torch
import torch.nn as nn

is_cuda = torch.cuda.is_available()


if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class SentimentNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size, hidden_dim, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim*2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.long()
        # embeds = self.embedding(x)
        # print('embeds', embeds)
        # lstm_out, hidden = self.lstm(embeds, hidden)
        # x = x.type(torch.LongTensor)
        lstm_out = self.lstm(x.float())

        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out[0])
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        return out


def train(model, train_loader, val_loader, epochs, optimizer, criterion):
    counter = 0
    print_every = 1
    clip = 5
    valid_loss_min = np.Inf

    model.train()
    for i in range(epochs):
        for inputs, mask, labels in train_loader:
            counter += 1

            inputs = inputs.type(torch.LongTensor)

            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()

            output = model(inputs)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if counter % print_every == 0:
                val_losses = []
                model.eval()
                for inp, mask, lab in val_loader:
                    inp, lab = inp.to(device), lab.to(device)
                    out = model(inp)
                    val_loss = criterion(out.squeeze(), lab.float())
                    val_losses.append(val_loss.item())

                model.train()
                print("Epoch: {}/{}...".format(i + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
                if np.mean(val_losses) <= valid_loss_min:
                    torch.save(model.state_dict(), './state_dict.pt')
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  '
                          'Saving model ...'.format(
                        valid_loss_min,
                        np.mean(val_losses)
                    ))
                    valid_loss_min = np.mean(val_losses)


def run(train_loader, val_loader, test_loader, epochs, INPUT_LENGTH):
    model = SentimentNet(
        input_size=INPUT_LENGTH,
        output_size=3,
        hidden_dim=256,
    )
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.0003

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
    )
    model.load_state_dict(torch.load('./state_dict.pt'))

    test_losses = []
    num_correct = 0

    model.eval()
    corrects = 0
    count_test = 0
    for inputs, mask, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        # test_loss = criterion(output.squeeze(), labels.float())
        # test_losses.append(test_loss.item())
        # pred = torch.round(output.squeeze())  # rounds the output to 0/1

        pred_class = torch.argmax(output, dim=1)
        labels_class = torch.argmax(labels, dim=1)
        corrects += torch.sum(pred_class == labels_class)
        count_test += 1

    print("Test loss: {:.3f}".format(np.mean(test_losses)))
    test_acc = corrects / count_test
    print("Test accuracy: {:.3f}%".format(test_acc * 100))
    print("corrects", corrects)
