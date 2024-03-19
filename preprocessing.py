import numpy as np


def get_words_and_tags(path, tagged):
    if tagged:
        file1 = open(path, encoding='utf-8')
        lines = file1.readlines()

        all_lines = []
        all_tags = []

        current_sentence = []
        current_tag = []

        for line in lines:
            if line[0] == "\t" or line[0] == "\n":
                all_lines.append(current_sentence)
                all_tags.append(current_tag)

                current_sentence = []
                current_tag = []

            else:
                word_tag = line.split("\t")
                word = word_tag[0]
                tag = word_tag[1].strip('\n')

                if word == '\ufeff':
                    continue

                else:
                    word = word.strip('\ufeff').strip('\n')
                    current_sentence.append(word)
                    current_tag.append(tag)

        X_tagged = np.array([all_lines[i][j] for i in range(len(all_lines)) for j in range(len(all_lines[i]))])
        y_tagged = np.array([0 if all_tags[i][j] == 'O' else 1 for i in range(len(all_tags))
                             for j in range(len(all_tags[i]))])

        return X_tagged, y_tagged

    else:
        file1 = open(path, encoding='utf-8')
        lines = file1.readlines()

        all_lines = []
        current_sentence = []

        for line in lines:
            if line[0] == "\t" or line[0] == "\n":
                all_lines.append(current_sentence)
                current_sentence = []

            else:
                word_tag = line.split("\t")
                word = word_tag[0]

                if word == '\ufeff':
                    continue

                else:
                    word = word.strip('\ufeff').strip('\n')
                    current_sentence.append(word)
        else:
            all_lines.append(current_sentence)

        X_untagged = np.array([all_lines[i][j] for i in range(len(all_lines)) for j in range(len(all_lines[i]))])

        return X_untagged


def get_sentences_and_tags(path, tagged):
    if tagged:
        file1 = open(path, encoding='utf-8')
        lines = file1.readlines()

        all_lines = []
        all_tags = []

        current_sentence = []
        current_tag = []

        for line in lines:
            if line[0] == "\t" or line[0] == "\n":
                all_lines.append(current_sentence)
                all_tags.append(current_tag)

                current_sentence = []
                current_tag = []

            else:
                word_tag = line.split("\t")
                word = word_tag[0]
                tag = word_tag[1].strip('\n')
                tag = 0 if tag == 'O' else 1

                if word == '\ufeff':
                    continue

                else:
                    word = word.strip('\ufeff').strip('\n')
                    current_sentence.append(word)
                    current_tag.append(tag)

        X_tagged = all_lines
        y_tagged = all_tags

        return X_tagged, y_tagged

    else:
        file1 = open(path, encoding='utf-8')
        lines = file1.readlines()

        all_lines = []
        current_sentence = []

        for line in lines:
            if line[0] == "\t" or line[0] == "\n":
                all_lines.append(current_sentence)
                current_sentence = []

            else:

                word_tag = line.split("\t")
                word = word_tag[0]

                if word == '\ufeff':
                    continue

                else:
                    word = word.strip('\ufeff').strip('\n')
                    current_sentence.append(word)

        return all_lines


def words2glove(X, y, glove_model):
    if y is not None:
        positive_tag_vectors = []
        negative_tag_vectors = []

        for word, tag in zip(X, y):
            if word in glove_model.key_to_index:
                vectorized_word = glove_model[word]

                if tag == 1:
                    positive_tag_vectors.append(vectorized_word)

                else:
                    negative_tag_vectors.append(vectorized_word)

        positive_tag_avg_vector = np.mean(positive_tag_vectors, axis=0)
        negative_tag_avg_vector = np.mean(negative_tag_vectors, axis=0)

        X_vectorised = []

        for word, tag in zip(X, y):
            features = add_features_to_embedding(word)
            word = word.lower()

            if word in glove_model.key_to_index:
                vectorized_word = glove_model[word]
                vectorized_word = np.concatenate((vectorized_word, features))
                X_vectorised.append(vectorized_word)

            else:
                if tag == 1:
                    vectorized_word = np.concatenate((positive_tag_avg_vector, features))

                else:
                    vectorized_word = np.concatenate((negative_tag_avg_vector, features))

                X_vectorised.append(vectorized_word)

    else:
        vectors_list = []

        for word in X:
            features = add_features_to_embedding(word)
            word = word.lower()

            if word in glove_model.key_to_index:
                vectorized_word = glove_model[word]
                vectorized_word = np.concatenate((vectorized_word, features))
                vectors_list.append(vectorized_word)

        avg_vector = np.mean(vectors_list, axis=0)

        X_vectorised = []

        for word in X:
            features = add_features_to_embedding(word)
            word = word.lower()

            if word in glove_model.key_to_index:
                vectorized_word = glove_model[word]
                vectorized_word = np.concatenate((vectorized_word, features))

            else:
                vectorized_word = avg_vector

            X_vectorised.append(vectorized_word)

    return np.array(X_vectorised)


def sentences2glove(X_padded, y_padded, glove_model):
    X_vectorized = []

    if y_padded is not None:
        positive_tag_vectors = []
        negative_tag_vectors = []
        j = 0

        for sen, tags in zip(X_padded, y_padded):
            for word, tag in zip(sen, tags):
                if word in glove_model.key_to_index:
                    vectorized_word = glove_model[word]

                    if tag == 1:
                        positive_tag_vectors.append(vectorized_word)

                    else:
                        negative_tag_vectors.append(vectorized_word)

        positive_tag_avg_vector = np.mean(positive_tag_vectors, axis=0)
        negative_tag_avg_vector = np.mean(negative_tag_vectors, axis=0)

        for sen, tags in zip(X_padded, y_padded):
            sentence_vectorized = []

            for word, tag in zip(sen, tags):
                features = add_features_to_embedding(word)
                word = word.lower()

                if word in glove_model.key_to_index:
                    vectorized_word = glove_model[word]
                    vectorized_word = np.concatenate((vectorized_word, features))
                    sentence_vectorized.append(vectorized_word)

                else:
                    if tag == 1:
                        vectorized_word = np.concatenate((positive_tag_avg_vector, features))
                        sentence_vectorized.append(vectorized_word)

                    elif tag == 0:
                        vectorized_word = np.concatenate((negative_tag_avg_vector, features))
                        sentence_vectorized.append(vectorized_word)

                    else:
                        vectorized_word = np.zeros(glove_model.vector_size)
                        vectorized_word = np.concatenate((vectorized_word, np.zeros(features.shape[0])))
                        sentence_vectorized.append(vectorized_word)

            for i in range(len(y_padded[j]), len(sen)):
                features = add_features_to_embedding(sen[i])
                vectorized_word = np.zeros(glove_model.vector_size)
                vectorized_word = np.concatenate((vectorized_word, np.zeros(features.shape[0])))
                sentence_vectorized.append(vectorized_word)

            j += 1
            X_vectorized.append(sentence_vectorized)

    else:
        vectors_list = []

        for sen in X_padded:
            for word in sen:
                features = add_features_to_embedding(word)
                word = word.lower()

                if word in glove_model.key_to_index:
                    vectorized_word = glove_model[word]
                    vectorized_word = np.concatenate((vectorized_word, features))
                    vectors_list.append(vectorized_word)

        avg_vector = np.mean(vectors_list, axis=0)

        X_vectorized = []

        for sen in X_padded:
            sentence_vectorized = []

            for word in sen:
                features = add_features_to_embedding(word)
                word = word.lower()

                if word in glove_model.key_to_index:
                    vectorized_word = glove_model[word]
                    vectorized_word = np.concatenate((vectorized_word, features))
                    sentence_vectorized.append(vectorized_word)

                elif word == '[pad]':
                    vectorized_word = np.zeros(glove_model.vector_size)
                    vectorized_word = np.concatenate((vectorized_word, np.zeros(features.shape[0])))
                    sentence_vectorized.append(vectorized_word)

                else:
                    vectorized_word = avg_vector
                    sentence_vectorized.append(vectorized_word)

            X_vectorized.append(sentence_vectorized)

    return np.array(X_vectorized)


def add_features_to_embedding(word):
    features_conditions = [lambda x: 1 if x[0].isupper() else 0,
                           lambda x: 1 if all(c.isascii() for c in x) else 0,
                           lambda x: 1 if x[0] == '@' else 0,
                           lambda x: 1 if any(c.isdigit() for c in x) else 0,
                           lambda x: 1 if all(c.isupper() for c in x) else 0,
                           lambda x: 1 if x[0:4] == 'http' else 0]

    features = []

    for f in features_conditions:
        features.append(f(word))

    return np.array(features)
