from itertools import cycle

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, auc

from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize

import itertools


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report


# crimedata=pd.read_csv(r'~/Workspace/ML_CW/CrimeDT.csv')
crimedata=pd.read_csv('Preprocessed_VideoGames.csv')
crimedata = crimedata.drop(crimedata.columns[0],axis=1)

categorise = {'Low': 0, 'Medium': 1, 'High': 2}
crimedata["GameTier"] = crimedata["GameTier"].map(categorise)

start = crimedata.columns.get_loc('Platform')
end = crimedata.columns.get_loc('User_normalised_by_year')
label = crimedata.columns.get_loc('GameTier')

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
    fig = plt.figure(figsize=(10, 8))
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


# Grid Search
from sklearn.model_selection import GridSearchCV

y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 14, test_size = 0.25)

# generate a random prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

parameter_space = {
    'hidden_layer_sizes': [(50,100,50),(50,50,50,50)],
    'max_iter': [1000],
    'activation': ['logistic'],
    'solver': ['adam'],
    # 'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','invscaling','adaptive'],
}

clf = RandomizedSearchCV(MLPClassifier(), parameter_space, n_jobs=-1)

# Best parameters found:
#  {'activation': 'logistic', 'hidden_layer_sizes': (50, 100, 50), 'max_iter': 1000, 'solver': 'adam'}

# clf = MLPClassifier(random_state=0, activation='logistic', hidden_layer_sizes=(50,100,50), max_iter=1000, solver= 'adam'learning_rate = 'adaptive')
clf.fit(X_train, y_train)

# clf = Perceptron(eta0=0.01, random_state=1, max_iter= 100)
# clf = CalibratedClassifierCV(clf)
# clf.fit(X_train, y_train)

print('Best parameters found:\n', clf.best_params_)


# Split dataset in to Train:Test - 75:25

# Instead of targets, store output as prediction probabilities
y_score = clf.predict_proba(X_test)

clf_predict = clf.predict(X_test)

# Accuracy factors
print('acc for training data: {:.3f}'.format(clf.score(X_train, y_train)))
print('acc for test data: {:.3f}'.format(clf.score(X_test, y_test)))
print('MLP Classification report:\n\n', classification_report(y_test, clf_predict))

cm = confusion_matrix(y_test.argmax(axis=1),clf_predict.argmax(axis=1))
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test.argmax(axis=1), clf_predict.argmax(axis=1))))
plot_confusion(cm,classes=["Low","Medium","High"])# disp.figure_.suptitle("Confusion Matrix")
# print("Confusion matrix:\n%s" % disp.confusion_matrix)

plt.show()


# ============================================================================
# ROC Curve Setup
# ============================================================================

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# ============================================================================
# Plot all ROC curves
# ============================================================================
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLP Crime Data ROC')
plt.legend(loc="lower right")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Zoomed in ROC')
plt.legend(loc="lower right")
plt.show()


