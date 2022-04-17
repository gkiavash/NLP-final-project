import csv
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")

txt_ = ''
txt_arr = []
with open("face_masks_train_retrieved.tsv", encoding='utf-8') as file:
    tsv_file = csv.reader(file, delimiter="\t")
    tokens = tokenizer(
        [line[5] for line in tsv_file],
        padding=True,
        truncation=True,
        return_tensors="np"
    )

    # tokens = tokenizer.encode_plus(
    #     tuple([line[5] for line in tsv_file]),
    #     max_length=128,
    #     truncation=True,
    #     padding='max_length',
    #     add_special_tokens=True,
    #     return_tensors='np'  # pt
    # )

    for line_index, line in enumerate(tsv_file):

        # TODO: all texts or perline?
        tokens = tokenizer.encode_plus(
            line[5],
            max_length=128,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='np'  # pt
        )

        print(tokens)


class BiLSTM(nn.Module):

    def __init__(self, linear_size, lstm_hidden_size, net_dropout, lstm_dropout):
        super(BiLSTM, self).__init__()

        self.model_name = 'BiLSTM'

        self.dropout = nn.Dropout(net_dropout)

        self.hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(1024, self.hidden_size, dropout=lstm_dropout, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 2, linear_size)
        self.out = nn.Linear(linear_size, 3)
        self.relu = nn.ReLU()

    def forward(self, x, x_len, epoch, target_word, _):
        x = x.squeeze(1)

        seq_lengths, perm_idx = x_len.sort(0, descending=True)
        seq_tensor = x[perm_idx, :, :]
        packed_input = pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input)
        _, unperm_idx = perm_idx.sort(0)
        h_t = ht[:, unperm_idx, :]
        h_t = torch.cat((h_t[0, :, :self.hidden_size], h_t[1, :, :self.hidden_size]), 1)

        linear = self.relu(self.linear(h_t))
        linear = self.dropout(linear)
        out = self.out(linear)

        return out

model = BiLSTM().to('cuda')
model(**tokens)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

model.train()
train_losses = []
for epoch in range(10):
    progress_bar = tqdm_notebook(train_loader, leave=False)
    losses = []
    total = 0
    for inputs, target in progress_bar:
        model.zero_grad()

        output = model(inputs)
        loss = criterion(output.squeeze(), target.float())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 3)

        optimizer.step()

        progress_bar.set_description(f'Loss: {loss.item():.3f}')

        losses.append(loss.item())
        total += 1

    epoch_loss = sum(losses) / total
    train_losses.append(epoch_loss)

    tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.3f}')