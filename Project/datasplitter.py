import pandas as pd
import sklearn



def concat(df1, name1, df2, name2):
    df1 = df1.assign(type=name1)
    df2 = df2.assign(type=name2)
    return pd.concat([df1, df2])


df1 = pd.read_csv('winequality-red.csv', delimiter=',')
df2 = pd.read_csv('winequality-white.csv', delimiter=',')

df = concat(df1, 0, df2, 1)
df = sklearn.utils.shuffle(df)

training_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(training_dataset.index)

training_target = training_dataset.pop('quality')
test_target = test_dataset.pop('quality')
print(test_target)

training_target.to_csv('training_target.csv')
test_target.to_csv('test_target.csv')

training_dataset.to_csv('training_data.csv')
test_dataset.to_csv('testing_data.csv')
