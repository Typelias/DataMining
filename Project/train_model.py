import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def concat(df1, name1, df2, name2):
    df1 = df1.assign(type=name1)
    df2 = df2.assign(type=name2)
    return pd.concat([df1, df2])


pd.set_option('display.max_columns', 15)

df1 = pd.read_csv('winequality-red.csv', delimiter=';')
df2 = pd.read_csv('winequality-white.csv', delimiter=';')

df = concat(df1, 0, df2, 1)
df.pop('type')

df['quality'] = (df['quality'] <= 5).astype(int)

print(df['quality'].value_counts())

print(df)

target = df.iloc[:, -1]
df = df[df.columns[:-1]]
# Normalize data
titles = df.columns.tolist()
pre = preprocessing.MinMaxScaler()
df = pd.DataFrame(pre.fit_transform(df), columns=titles)

x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=13)

test_data = x_test

test_data = test_data.assign(quality=y_test.array)
test_data.to_csv('test_data.csv', ';', index=False)

# print(y_train)
# print('------------------------------------------------------------')
# print(x_train)

# model = keras.Sequential([
#     layers.Dense(256, activation='relu', input_dim=11),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(
#     x_train,
#     y_train,
#     epochs=50,
#     batch_size=10,
#     verbose=2
# )
# model.save('model.h5')
