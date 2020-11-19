import random
import numpy as np
from cs231n.data_utils import load_CIFAR10

cifar10_dir = 'D:\\code\\assignment1\\cs231n\\datasets\\cifar-10-batches-py'
try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []

avg_size=int(X_train.shape[0]/num_folds)
for i in range(num_folds):
    X_train_folds.append(X_train[i*avg_size:(i+1)*avg_size])
    y_train_folds.append(y_train[i*avg_size:(i+1)*avg_size])

for k in k_choices:
    for i in range(num_folds):
        X_train_cv=np.vstack(X_train_folds[0:i]+X_train_folds[i+1:])
        #print(y_train_folds.shape)
        y_train_cv=np.hstack(y_train_folds[0:i]+y_train_folds[i+1:])
        X_valid_cv=X_train_folds[i]
        y_valid_cv=y_train_folds[i]