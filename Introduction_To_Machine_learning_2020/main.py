import regec2 as r
import numpy as np

train_file = "/Users/ivangrechikhin/Documents/Datasets/Banknote/data.csv"
test_file = "/Users/ivangrechikhin/Documents/Datasets/Banknote/test.csv"

train = np.loadtxt(train_file, delimiter=',')
test = np.loadtxt(test_file, delimiter=',')

train_l = train[:, -1]
test_l = test[:, -1]
train = train[:, 0:-1]
test = test[:, 0:-1]

[class_l, acc, Z, W] = r.regec(train, train_l, test, test_l, 900, 0.1, 0.1)
print(acc)
