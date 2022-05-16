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
        self.dropout = nn.Dropout(0.2)
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
        for inputs, labels in train_loader:
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
                for inp, lab in val_loader:
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
