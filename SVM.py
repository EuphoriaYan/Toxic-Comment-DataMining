import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR, NuSVR, LinearSVR

from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/train_preprocessed.csv')
test = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test_preprocessed.csv')
submission = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
test_label = pd.read_csv('./input/jigsaw-toxic-comment-classification-challenge/test_labels.csv')

X_train = train["comment_text"].fillna("fillna").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train[list_classes].values
# X_test = test["comment_text"].fillna("fillna").values

X_features = np.load("./model/feature_matrix.npy")

X_features = np.array(X_features, dtype=np.float)

X_features = preprocessing.scale(X_features)

# min_max_scaler = preprocessing.MinMaxScaler()
# X_features= min_max_scaler.fit_transform(X_features)

y_train = np.array(y_train, dtype=np.int)

# np.savetxt("1.txt",y_train)

# y_train = MultiLabelBinarizer().fit_transform(y_train)

# np.savetxt("2.txt",y_train)

X_tra, X_test, y_tra, y_test = train_test_split(X_features, y_train, train_size=0.95, random_state=233)

classifier = SVC(gamma='auto', kernel='rbf', max_iter=300)
# cls = DecisionTreeClassifier()
classifier = OneVsRestClassifier(classifier)

# train
classifier.fit(X_tra, y_tra)

# predict
predictions = classifier.predict(X_test)
score=roc_auc_score(y_test,predictions)

print("\n ROC-AUC - score: %.6f \n" % score)
