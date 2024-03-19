import preprocessing as pp
from FFNmodel import FFN, train_FFN, test_FFN, predict_FFN
from torch.optim import Adam
from gensim import downloader


def main():
    GLOVE_PATH = 'glove-twitter-200'
    train_path = 'train.tagged'
    val_path = 'dev.tagged'
    test_path = 'test.untagged'

    # Downloading glove
    print("Downloading GloVe...")
    glove_model = downloader.load(GLOVE_PATH)
    print("GloVe downloaded successfully !\n")

    # Preprocessing
    print("Preprocessing...")
    X_train, y_train = pp.get_words_and_tags("train.tagged", True)

    X_validation, y_validation = pp.get_words_and_tags("dev.tagged", True)

    X_test = pp.get_words_and_tags("test.untagged", False)

    X_train = pp.words2glove(X_train, y_train, glove_model)
    X_validation = pp.words2glove(X_validation, y_validation, glove_model)
    X_test = pp.words2glove(X_test, None, glove_model)

    print("Preprocessing is done !\n")

    batch_size = 16
    num_epochs = 50

    model_2 = FFN(input_dimension=len(X_train[0]), hidden_dimension=100, num_classes=2)
    optimizer = Adam(params=model_2.parameters())

    train_FFN(model_2, X_train, y_train, optimizer, batch_size, num_epochs)
    print("test on train:\n")
    test_FFN(model_2, X_train, y_train, batch_size)
    print("test on validation:\n")
    test_FFN(model_2, X_validation, y_validation, batch_size)

    # Test part on the test.untagged file and creation of the predicted_file
    predict_FFN(model_2, X_test, optimizer, batch_size, num_epochs, 'comp_342791324_931214522.tagged')


if __name__ == '__main__':
    main()
