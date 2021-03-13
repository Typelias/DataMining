from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the data
dataSet = arff.loadarff("breast-cancer.arff")
# Convert the data to pandas dataframe
df = pd.DataFrame(dataSet[0])
# Go through the dataframe and decode it to remove weird characters
for i in range(0, len(df.columns)):
    title = list(df.columns)[i]
    df[title] = df[title].apply(lambda s: s.decode("utf-8"))

# Make a copy of the column names and class names for later
feature_names = list(df.columns)
class_names = df['Class'].unique()

# Remove rows with nan values
df = df[df['node-caps'] != '?']
df = df[df['breast-quad'] != '?']
# Label encode the whole dataframe
df = df.apply(LabelEncoder().fit_transform)

# Remove the Class from the dataframe
Class = df.iloc[:, -1]
df = df[df.columns[:-1]]

# Split the data to a training set and a testing set
x_train, x_test, y_train, y_test = train_test_split(df, Class, test_size=0.2, random_state=0)

# Naive Bayes classification
# Create an instance of Naive Bayes
gnb = GaussianNB()
# Fit the model to the training data and predict the testing data
y_pred = gnb.fit(x_train, y_train).predict(x_test)

# Print out how many wrong classifications it did
print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
# Calculate the accuracy
correct = (y_test == y_pred).sum()
accuracy = correct / len(y_pred)
# Printing the accuracy
print('Accuracy NB:', accuracy * 100)

# Confusion Matrix
names = ['recurr', 'no-recurr']
metrics.plot_confusion_matrix(gnb, x_test, y_test, display_labels=names)

# Decision Tree classification
# Create an instance of a Decision Tree
clf = tree.DecisionTreeClassifier(criterion='gini', ccp_alpha=0.0075)
# Fit the model to the training data and predict the testing data
clf = clf.fit(x_train, y_train)
# Calculate the accuracy
tree_pred = clf.predict(x_test)
corr = (y_test == tree_pred).sum()
accuracy_tree = corr / len(tree_pred)
# Print out how many wrong classifications it did
print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != tree_pred).sum()))
# Printing the accuracy
print('Accuracy Tree:', accuracy_tree * 100)

# Matrix

metrics.plot_confusion_matrix(clf, x_test, y_test, display_labels=names)

# ROC Curve
fpr = dict()
tpr = dict()
roc_auc = dict()
y_true = y_test.to_numpy()

fpr['NB'], tpr['NB'], _ = roc_curve(y_true, y_pred)
roc_auc['NB'] = auc(fpr['NB'], tpr['NB'])

fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

tnN, fpN, fnN, tpN = metrics.confusion_matrix(y_true, y_pred).ravel()
tnT, fpT, fnT, tpT = metrics.confusion_matrix(y_true, tree_pred).ravel()

# True positive

true_positive_NB = tpN / (tpN + fnN)
true_positive_DT = tpT / (tpT + fnT)

# True negative
true_negative_NB = tnN / (tnN + fpN)
true_negative_DT = tnT / (tnT + fpT)

print('True positive rate NB:\t', str(round(true_positive_NB * 100, 3)) + '%')
print('True negative rate NB:\t', str(round(true_negative_NB * 100, 3)) + '%')
print('True positive rate DT:\t', str(round(true_positive_DT * 100, 3)) + '%')
print('True negative rate DT:\t', str(round(true_negative_DT * 100, 3)) + '%')

# Plot the roc curve
plt.figure()
lw = 2
plt.plot(fpr['NB'], tpr['NB'], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc['micro'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('roc.png')
plt.show()

# Plotting the Decision Tree
plt.figure()
fig = plt.figure(figsize=(40, 35))
tree.plot_tree(clf, filled=True, feature_names=feature_names, class_names=class_names)
plt.savefig('tree.png')
# plt.show()
