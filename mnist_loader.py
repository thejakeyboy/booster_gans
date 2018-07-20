import numpy as np
import torch
from torchvision import transforms, datasets


DATA_FOLDER = './torch_data/DCGAN/MNIST'

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


def get_mnist_9_4():

    data = mnist_data()

    X = data.train_data[np.logical_or(data.train_labels == 9, data.train_labels==4),:]
    Y = data.train_labels[np.logical_or(data.train_labels == 9, data.train_labels==4)] == 9
    X,Y = X.numpy(), Y.numpy()


    indices = np.array(range(X.shape[0]))
    np.random.shuffle(indices)
    num_train = int(0.75 * len(indices))
    train_indices, test_indices = indices[:num_train], indices[num_train:]

    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]

    return X_train, Y_train, X_test, Y_test