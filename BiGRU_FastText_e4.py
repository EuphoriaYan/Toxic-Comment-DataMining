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


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPool1D()(x)
    conc = concatenate([avg_pool, max_pool])
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(conc)
    x = Dropout(0.1)(x)
    out = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


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
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    '''
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

    embedding_matrix = np.load('./model/embedding_matrix.npy')

    model = get_model()

    batch_size = 32
    epochs = 4

    X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)
    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
    model.fit(X_tra, y_tra,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              callbacks=[RocAuc],
              verbose=2)

    model.save('./model/BiGRU-FastText-e4.hdf5')

    y_pred = model.predict(x_test)
    y_pred = [[1 if score > 0.5 else 0 for score in case] for case in y_pred]
    y_labels = test_label[list_classes].values

    submission[list_classes] = y_pred
    submission.to_csv('./output/BiGRU-FastText-e4.csv', index=False)

    total = 0
    correct = 0

    for (pred, label) in zip(y_pred, y_labels):
        for i in range(6):
            if label[i] == -1:
                continue
            total += 1
            if label[i] == pred[i]:
                correct += 1

    print("total = %d" % total)
    print("correct = %d" % correct)
    print("acc = %.4f" % (1.0*correct/total))
