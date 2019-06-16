import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import Bidirectional, GRU, GlobalMaxPool1D, GlobalAveragePooling1D, SpatialDropout1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'


def get_word_embedding_model():
    max_features = 30000
    maxlen = 100
    embedding_size = 300
    embedding_matrix = np.load('./model/embedding_matrix.npy')

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix])(inp)
    # TODO

    model = Model(inputs=inp, outputs=x)  # TODO
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_sentence_embedding_model():
    max_features = 30000
    maxlen = 100
    embedding_size = 300
    embedding_matrix = np.load('./model/embedding_matrix.npy')

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix])(inp)
    x = GlobalAveragePooling1D()(x)

    model = Model(inputs=inp, outputs=x)

    return model


def get_sentence_features(X_train, X_test):
    max_features = 30000
    maxlen = 100

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

    model = get_sentence_embedding_model()

    x_feature = model.predict(x_train)

    return x_feature


if __name__ == "__main__":
    train = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/train_preprocessed.csv')
    test = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test_preprocessed.csv')
    submission = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
    test_label = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')

    X_train = train["comment_text"].fillna("fillna").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_train = train[list_classes].values
    X_test = test["comment_text"].fillna("fillna").values

    X_features = get_sentence_features(X_train, X_test)
    np.save("./model/feature_matrix.npy", X_features)
