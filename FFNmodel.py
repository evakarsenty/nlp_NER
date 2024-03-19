import torch
from abc import ABC
from torch import nn
import preprocessing as pp
from torch.utils.data import TensorDataset, DataLoader, Dataset
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class FFN(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_classes):
        super(FFN, self).__init__()
        self.first_layer = nn.Linear(input_dimension, hidden_dimension)
        self.second_layer = nn.Linear(hidden_dimension, num_classes)

        self.first_activation = nn.LeakyReLU()
        self.second_activation = nn.Softmax()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, X, labels):
        X = self.first_layer(X)
        X = self.first_activation(X)

        X = self.second_layer(X)
        y_pred = self.second_activation(X)

        if labels is None:
            return y_pred, None

        loss = self.loss(labels, y_pred)
        return y_pred, loss


def one_hot(y, num_of_classes=2):
    y = torch.tensor(y)
    hot = torch.zeros((y.size()[0], num_of_classes))
    hot[torch.arange(y.size()[0]), y] = 1
    return hot


def calculate_confusion_measures(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0

    for label1, label2 in zip(y_true, y_pred):
        if label1 == 1 and label2 == 1:
            tp += 1

        elif label1 == 1 and label2 == 0:
            fn += 1

        elif label1 == 0 and label2 == 1:
            fp += 1

        else:
            tn += 1

    return [tp, tn, fp, fn]


def calculate_f1_score(tp, tn, fp, fn):
    epsilon = 1e-7
    # in order to avoid value 0 on denominator

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_score


class FFNDataSetTest(Dataset, ABC):

    def __init__(self, sentences):
        self.X = sentences

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item]


def train_FFN(model, X_train, y_train, optimizer, batch_size, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        tp, tn, fp, fn = 0, 0, 0, 0

        for glove_words, labels in train_data_loader:
            batch_glove_words = glove_words.to(device)
            tags = (one_hot(labels, 2)).to(device)

            optimizer.zero_grad()
            outputs, loss = model.forward(batch_glove_words, tags)
            loss.backward()
            optimizer.step()

            predicted = [torch.argmax(scores).item() for scores in outputs]

            confusion_measures = calculate_confusion_measures(labels, predicted)

            tp += confusion_measures[0]
            tn += confusion_measures[1]
            fp += confusion_measures[2]
            fn += confusion_measures[3]

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_score = calculate_f1_score(tp, tn, fp, fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print(f"f1 score for epoch {epoch + 1}: {f1_score}")
        print(f"accuracy for epoch {epoch + 1}: {accuracy}")
        print(f"recall for epoch {epoch + 1}: {recall}")
        print(f"precision for epoch {epoch + 1}: {precision}")


def test_FFN(model, X_test, y_test, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_test, y_test)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tp, tn, fp, fn = 0, 0, 0, 0

    if y_test is not None:
        for glove_words, labels in test_data_loader:
            batch_glove_words = glove_words.to(device)
            tags = (one_hot(labels, 2)).to(device)

            with torch.no_grad():
                outputs, loss = model.forward(batch_glove_words, tags)

            predicted = (torch.tensor([torch.argmax(scores).item() for scores in outputs])).to(device)
            confusion_measures = calculate_confusion_measures(labels, predicted)

            tp += confusion_measures[0]
            tn += confusion_measures[1]
            fp += confusion_measures[2]
            fn += confusion_measures[3]

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_score = calculate_f1_score(tp, tn, fp, fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print(f"f1 score: {f1_score}")
        print(f"accuracy: {accuracy}")
        print(f"recall: {recall}")
        print(f"precision: {precision}")


def predict_FFN(model, X_test, optimizer, batch_size, num_epochs, output_file):
    seed = 44
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    predicted_tags = []
    test_data = pp.get_sentences_and_tags('test.untagged', False)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    test_dataset = FFNDataSetTest(X_test)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for glove_words in test_data_loader:
        batch_glove_words = glove_words.to(device)
        optimizer.zero_grad()
        outputs, loss = model.forward(batch_glove_words, None)
        optimizer.step()

        predicted = [torch.argmax(scores).item() for scores in outputs]
        predicted_tags.append(predicted)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sen, tags in zip(test_data, predicted_tags):
            for word, tag in zip(sen, tags):
                if tag == 0:
                    f.write(f"{word}\t{'O'}\n")
                else:
                    f.write(f"{word}\t{'1'}\n")
            f.write("\n")

    print("The predictions file was created successfully ! ")
