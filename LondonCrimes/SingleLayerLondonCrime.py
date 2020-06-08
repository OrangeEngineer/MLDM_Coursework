from itertools import cycle

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, auc, mean_absolute_error, classification_report, confusion_matrix

from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
import itertools


# crimedata=pd.read_csv(r'~/Workspace/ML_CW/CrimeDT.csv')
crimedata=pd.read_csv('Preprocessed_LodonCrime.csv')
crimedata = crimedata.drop(crimedata.columns[0],axis=1)

categorise = {'Low': 0, 'Medium': 1, 'High': 2}
crimedata["CrimeTier"] = crimedata["CrimeTier"].map(categorise)

start = crimedata.columns.get_loc('MinorCategory')
end = crimedata.columns.get_loc('2018')
label = crimedata.columns.get_loc('CrimeTier')

X = crimedata.values[:,start:end]
enc = preprocessing.OrdinalEncoder()
enc.fit(X)
X = enc.transform(X)
print (X)

# Extracting target/ class labels
y = crimedata.values[:,label].astype(float)
print (y)

######################################################################################################

####################         Multilayer Perceptron Part         ######################################

######################################################################################################


def plot_confusion(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=(13, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 14, test_size = 0.20)
# generate a random prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]


clf = Perceptron(eta0=0.1, random_state=0, max_iter=1000)
clf.fit(X_train, y_train.argmax(axis=1))


# Split dataset in to Train:Test - 75:25

# Instead of targets, store output as prediction probabilities
y_score = clf.predict(X_test)

clf_predict = clf.predict(X_test)
clf_predict_on_train = clf.predict(X_train)
# Accuracy factors
print('acc for training data: {:.3f}'.format(clf.score(X_train, y_train.argmax(axis=1))))
print('acc for test data: {:.3f}'.format(clf.score(X_test, y_test.argmax(axis=1))))
print('MLP Classification report:\n\n', classification_report(y_test.argmax(axis=1), clf_predict))


# disp = metrics.plot_confusion_matrix(clf, X_test, y_test.argmax(axis=1))
# disp.figure_.suptitle("Confusion Matrix")
# print("Confusion matrix:\n%s" % disp.confusion_matrix)
# #
# plt.show()
#
#
cm = confusion_matrix(y_test.argmax(axis=1),clf_predict)
cm_on_train = confusion_matrix(y_train.argmax(axis=1),clf_predict_on_train)
# print(cm)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test.argmax(axis=1), clf_predict)))
plot_confusion(cm,classes=["Low","Medium","High"], title='Test Set Confusion matrix')# disp.figure_.suptitle("Confusion Matrix")
# print("Confusion matrix:\n%s" % disp.confusion_matrix)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_train.argmax(axis=1), clf_predict_on_train)))
plot_confusion(cm_on_train,classes=["Low","Medium","High"], title='Training Set Confusion matrix')# disp.figure_.suptitle("Confusion Matrix")
# print("Confusion matrix:\n%s" % disp.confusion_matrix)
plt.show()
