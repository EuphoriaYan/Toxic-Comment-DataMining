import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint


def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(GRU(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    max_features = 20000
    maxlen = 100

    train = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/train.csv')
    test = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test.csv')
    test_label = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("NULL").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("NULL").values

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_tr = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    model = get_model()
    file_path = "./model/weights_base.best.hdf5"

    batch_size = 32
    epochs = 2

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

    callbacks_list = [checkpoint, early]  # early
    model.fit(X_tr, y,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              callbacks=callbacks_list)

    model.load_weights(file_path)

    y_pred = model.predict(X_te)
    y_pred = [[1 if score > 0.5 else 0 for score in case] for case in y_pred]

    y_labels = test_label[list_classes].values

    '''
    sample_submission = pd.read_csv("./jigsaw-toxic-comment-classification-challenge/sample_submission.csv")
    sample_submission[list_classes] = y_test
    sample_submission.to_csv("GRU-Baseline.csv", index=False)
    '''

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
