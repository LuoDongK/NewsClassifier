import pandas as pd
from torch import nn, optim
from torchtext import data
from torchtext.data import Iterator
import torch
import numpy as np
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_classes):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out


class NLPDataLoader(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, data_set, text_field, label_field, test=False):
        self.data = data_set
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        if test:
            for text in data_set['text']:
                examples.append(data.Example.fromlist([text.split(' '), 0], fields))
        else:
            for text, label in zip(data_set['text'], data_set['label']):
                examples.append(data.Example.fromlist([text.split(' '), label], fields))
        super(NLPDataLoader, self).__init__(examples, fields)


def accuracy(predictions, label):
    predictions = torch.argmax(predictions, dim=1)
    correct = torch.eq(predictions, label).float()
    acc = correct.sum() / len(correct)
    return acc


def data_split(text_field, label_field, dataset, mode=False):
    if mode == 'init':
        for index, c in enumerate(dataset):
            partial = NLPDataLoader(c, text_field=text_field, label_field=label_field, test=False)
            if index == 0:
                text_field.build_vocab(partial)
                label_field.build_vocab(list(range(13)))
            else:
                text_counter = text_field.vocab.freqs
                for example in partial.examples:
                    text_counter.update(example.text)
                text_field.vocab = text_field.vocab_cls(text_counter, specials=['<unk>', '<pad>'])
        return
    elif mode is False:
        dataset = NLPDataLoader(dataset, text_field=text_field, label_field=label_field, test=False)
        return Iterator.splits((dataset,), batch_size=20)
    elif mode is True:
        dataset = NLPDataLoader(dataset, text_field=text_field, label_field=label_field, test=True)
        return Iterator.splits((dataset,), batch_size=20, shuffle=False)


def build_field(stop_word):
    for ii in range(len(stop_word)):
        stop_word[ii] = str(stop_word[ii])
    text_field = data.Field(stop_words=stop_word)
    label_field = data.LabelField(use_vocab=False)
    return text_field, label_field


class NewsClassifier(object):
    def __init__(self, model):
        torch.manual_seed(123)
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.best_acc, self.best_epoch = 0, 0
        self.validate_accuracy = []
        self.predictions = []

    def train(self, batch_text, batch_label):
        self.model.train()
        output = self.model(batch_text)
        loss = self.criterion(output, batch_label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def validate(self, batch_text, batch_label):
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch_text)
            acc = accuracy(output, batch_label).item()

        self.validate_accuracy.append(acc)

    def predict(self, batch_text):
        self.model.eval()
        with torch.no_grad():
            predictions = torch.argmax(self.model(batch_text), dim=1)
            self.predictions.extend(list(predictions.cpu().detach().numpy()))


# 初始化

device = torch.device('cuda')
textField, labelField = build_field(list(pd.read_csv('stopwords.csv')['words']))

print('building vocabulary')
data_iter = pd.read_csv('stratified.csv', chunksize=500)
data_split(textField, labelField, data_iter, 'init')

print('initializing')
rnn = RNN(len(textField.vocab), 50, 80, 14)
rnn = rnn.to(device)
classifier = NewsClassifier(rnn)

# 训练

for epoch in range(20):
    classifier.validate_accuracy = []
    print('Epoch {} is processing'.format(epoch))
    data_iter = pd.read_csv('stratified.csv', chunksize=200)
    for chunk_index, chunk in enumerate(data_iter):
        if chunk_index % 100 == 0:
            print(chunk_index)
        if chunk_index < 800:
            train_iter = data_split(textField, labelField, chunk)[0]
            for batch in train_iter:
                classifier.train(batch.text.to(device), batch.label.to(device))
        else:
            validate_iter = data_split(textField, labelField, chunk)[0]
            for batch in validate_iter:
                classifier.validate(batch.text.to(device), batch.label.to(device))
    avg_acc = np.array(classifier.validate_accuracy).mean()
    if avg_acc > classifier.best_acc:
        classifier.best_epoch = epoch
        classifier.best_acc = avg_acc
        torch.save(classifier.model.state_dict(), '50*96.mdl')
    print('第{}代准确率:{}'.format(epoch, avg_acc))
print('The best accuracy {} was obtained in epoch {}'.format(classifier.best_acc, classifier.best_epoch))

# %% md

# 预测

rnn.load_state_dict(torch.load('50*80.mdl'))
classifier.predictions = []
data_iter = pd.read_csv('test.csv', chunksize=500)
for chunk in data_iter:
    test_iter = data_split(textField, labelField, chunk, True)[0]
    for batch in test_iter:
        classifier.predict(batch.text.to(device))
pd.DataFrame(classifier.predictions).to_csv('predictions.csv', index=False, header=['label'])

# 可视化

with open('train_accuracy.txt', 'r') as f:
    train_accuracy = f.read()
    train_accuracy = [float(item) for item in train_accuracy.split()]
    temp = []
    for s in range(40):
        temp.append(np.array(train_accuracy[s * 4000:(s + 1) * 4000]).mean())
    train_accuracy = temp

with open('validate_accuracy.txt', 'r') as f:
    validate_accuracy = f.read()
    validate_accuracy = [float(item) for item in validate_accuracy.split()]
    temp = []
    for s in range(40):
        temp.append(np.array(validate_accuracy[s * 1000:(s + 1) * 1000]).mean())
    validate_accuracy = temp

plt.figure(figsize=[10, 5])
plt.plot(train_accuracy)
plt.plot(validate_accuracy)
plt.legend(labels=['train_accuracy', 'validate_accuracy'])
plt.show()