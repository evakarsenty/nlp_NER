from torch.optim import Adam
import preprocessing as pp
import torch
from torch.utils.data import DataLoader
from gensim import downloader
from SVMmodel import train_SVM, test_SVM, calculate_f1_score
from FFNmodel import FFN, train_FFN, test_FFN, predict_FFN
from LSTMmodel import MyLSTM, train_LSTM, LSTMDataSet, padding_, tokenize, test_LSTM, LSTMDataSetTest
import numpy as np

GLOVE_PATH = 'glove-twitter-200'
train_path = 'train.tagged'
val_path = 'dev.tagged'
test_path = 'test.untagged'


def main():
    # Downloading glove
    print("Downloading GloVe...")
    glove_model = downloader.load(GLOVE_PATH)
    print("GloVe downloaded successfully !\n")

    # Preprocessing
    print("Preprocessing...")
    X_train, y_train = pp.get_words_and_tags("train.tagged", True)
    X_train_LSTM, y_train_LSTM = pp.get_sentences_and_tags("train.tagged", True)

    X_validation, y_validation = pp.get_words_and_tags("dev.tagged", True)
    X_validation_LSTM, y_validation_LSTM = pp.get_sentences_and_tags("dev.tagged", True)

    X_test = pp.get_words_and_tags("test.untagged", False)
    X_test_LSTM = pp.get_sentences_and_tags("test.untagged", False)

    X_train = pp.words2glove(X_train, y_train, glove_model)
    X_validation = pp.words2glove(X_validation, y_validation, glove_model)
    X_test = pp.words2glove(X_test, None, glove_model)

    X_train_LSTM, X_validation_LSTM, X_test_LSTM, word2idx, idx2word = tokenize(X_train_LSTM, X_validation_LSTM,
                                                                                X_test_LSTM)

    vocab_size = len(word2idx)

    train_sentence_lens = [min(len(s), 300) for s in X_train_LSTM]
    validation_sentence_lens = [min(len(s), 300) for s in X_validation_LSTM]
    test_sentence_lens = [min(len(s), 300) for s in X_test_LSTM]

    X_train_LSTM = padding_(X_train_LSTM, 300, False)
    y_train_LSTM = padding_(y_train_LSTM, 300, True)

    X_validation_LSTM = padding_(X_validation_LSTM, 300, False)
    y_validation_LSTM = padding_(y_validation_LSTM, 300, True)

    X_test_LSTM = padding_(X_test_LSTM, 300, False)

    print("Preprocessing is done !\n")

    print("*" * 20 + " Model 1 " + "*" * 30)

    model_1 = train_SVM(X_train, y_train)
    y_pred_train = test_SVM(model_1, X_train)
    y_pred_validation = test_SVM(model_1, X_validation)

    f1_score_train = calculate_f1_score(y_train, y_pred_train)
    f1_score_validation = calculate_f1_score(y_validation, y_pred_validation)

    print(f"Train set: f1 score = {f1_score_train:.4f}")
    print(f"Validation set: f1 score = {f1_score_validation:.4f}\n")

    print("*" * 20 + " Model 2 " + "*" * 30)

    batch_size = 16
    num_epochs = 50

    model_2 = FFN(input_dimension=len(X_train[0]), hidden_dimension=100, num_classes=2)
    optimizer = Adam(params=model_2.parameters())

    train_FFN(model_2, X_train, y_train, optimizer, batch_size, num_epochs)
    print("test on train:\n")
    test_FFN(model_2, X_train, y_train, batch_size)
    print("test on validation:\n")
    test_FFN(model_2, X_validation, y_validation, batch_size)
    predict_FFN(model_2, X_test, optimizer, batch_size, num_epochs, 'comp_342791324_931214522.tagged')

    print("*" * 20 + " Model 3 " + "*" * 30)

    seed = 44
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = LSTMDataSet(X_train_LSTM, train_sentence_lens, y_train_LSTM)
    validation_dataset = LSTMDataSet(X_validation_LSTM, validation_sentence_lens, y_validation_LSTM)
    test_dataset = LSTMDataSetTest(X_test_LSTM, test_sentence_lens)

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    model3 = MyLSTM(vocab_size, tag_dim=2)
    model3.to(device)
    optimizer = torch.optim.Adam(model3.parameters(), lr=0.005)

    best_f1 = 0
    best_epoch = 0

    # Train part
    for epoch in range(1000):
        print(f"\n -- Epoch {epoch} --")
        f1 = train_LSTM(model3, device, optimizer, train_dataloader, validation_dataloader)

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch

        if epoch - best_epoch == 15:
            break

    print(f"best validation f1 score : {best_f1:.2f} in epoch {best_epoch}")
    test_LSTM(device, model3, test_dataloader, 'predictions.tagged')


if __name__ == '__main__':
    main()
