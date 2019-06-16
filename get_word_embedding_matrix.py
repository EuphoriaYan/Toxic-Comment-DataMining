from keras.preprocessing import text, sequence
import numpy as np
import pandas as pd


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


if __name__ == '__main__':

    train = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/train_preprocessed.csv')
    test = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test_preprocessed.csv')
    submission = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
    test_label = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')

    X_train = train["comment_text"].fillna("fillna").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_train = train[list_classes].values
    X_test = test["comment_text"].fillna("fillna").values

    max_features = 30000
    maxlen = 100
    embedding_size = 300

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    last_w = tokenizer.index_word[9999]
    print(last_w)
    print(tokenizer.word_counts[last_w])
    last_w = tokenizer.index_word[19999]
    print(last_w)
    print(tokenizer.word_counts[last_w])
    last_w = tokenizer.index_word[29999]
    print(last_w)
    print(tokenizer.word_counts[last_w])


'''
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    embedding_file = './model/crawl-300d-2M.vec'

    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' '))
                            for o in open(embedding_file, encoding='utf-8', errors='ignore'))

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embedding_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    np.save("./model/embedding_matrix.npy", embedding_matrix)
'''
