import csv
import numpy as np
from collections import Counter

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import preprocess

tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")


with open("face_masks_train_retrieved.tsv", encoding='utf-8') as file:
    tsv_file = csv.reader(file, delimiter="\t")
    comments = []
    labels = []
    for line_index, line in enumerate(tsv_file):
        if line_index == 0:
            continue
        comments.append(preprocess(line[5]))
        labels.append(line[4])


tokens = tokenizer(
    comments,
    padding='max_length',
    max_length=128,
    truncation=True,
    add_special_tokens=True,
    return_tensors="tf"
)
print(tokens)

print(labels)
count_words = Counter(labels)
print(count_words.keys())
print(count_words.values())

print("[INFO] class labels:")
integer_encoded = LabelEncoder().fit_transform(np.array(labels))
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = OneHotEncoder(sparse=False).fit_transform(integer_encoded)
print(onehot_encoded)
#
# from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
#
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
#
# from transformers import DataCollatorWithPadding
#
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
#
# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=5,
#     weight_decay=0.01,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_imdb["train"],
#     eval_dataset=tokenized_imdb["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
# )
#
# trainer.train()
#
