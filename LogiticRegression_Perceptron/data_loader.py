import json
import numpy as np

# Hand-written digits data
def data_loader_mnist(dataset='mnist_subset.json'):
  # This function reads the MNIST data and separate it into train, val, and test set
  with open(dataset, 'r') as f:
        data_set = json.load(f)
  train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

  return np.asarray(train_set[0]), \
          np.asarray(test_set[0]), \
          np.asarray(train_set[1]), \
          np.asarray(test_set[1])



