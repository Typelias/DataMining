import pandas as pd
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

test_data = pd.read_csv('test_data.csv', delimiter=';')

# print(test_data)
target = test_data.iloc[:, -1]
df = test_data[test_data.columns[:-1]]
df.info()

model = keras.models.load_model('model.h5')

# model.summary()
eval = model.evaluate(df, target, return_dict=True)
print(eval)
# print('Mean Square Error', eval['mse'])
prediction = model.predict(df)
prediction = prediction.round().tolist()
prediction = [item for sublist in prediction for item in sublist]
# target = target.to_list()
# corr = 0
# for i in range(len(prediction)):
#     if prediction[i] == target[i]:
#         corr += 1
# acc = corr / len(prediction)
# print('Number of correct guesses:', corr)
# print('Accuracy', acc)
matrix = confusion_matrix(target, prediction)
plt_df = pd.DataFrame(matrix)
tn, fp, fn, tp = matrix.ravel()

tpr = tp / (tp + fp)

tnr = tn / (tn + fp)

print('True Positive Rate:', tpr)
print('True negative Rate:', tnr)

plt.figure(figsize=(10, 7))
sn.heatmap(plt_df, annot=True, fmt='g')

plt.show()
