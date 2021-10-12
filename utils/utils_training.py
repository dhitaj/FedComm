import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import wiki_noniid, Corpus, batchify


def get_dataset(dataset, iid, unequal, num_users):
    ntokens = 0
    if dataset == 'cifar10':
        data_dir = 'data/cifar10/'

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        if iid:
            user_groups = cifar_iid(train_dataset, num_users, dataset)
        else:
            if unequal:
                raise NotImplementedError()
            else:
                user_groups = cifar_noniid(train_dataset, num_users, dataset)

    elif dataset == 'mnist':
        data_dir = 'data/mnist/'

        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        if iid:
            user_groups = mnist_iid(train_dataset, num_users, dataset)
        else:
            if unequal:
                user_groups = mnist_noniid_unequal(train_dataset, num_users, dataset)
            else:
                user_groups = mnist_noniid(train_dataset, num_users, dataset)

    elif dataset == 'wiki':
        data_dir = 'data/wiki/'
        corpus = Corpus(data_dir)

        train_dataset = batchify(corpus.train, 20)
        test_dataset = batchify(corpus.test, 20)
        user_groups = wiki_noniid(train_dataset, num_users, dataset)

        return train_dataset, test_dataset, user_groups, len(corpus.dictionary)

    return train_dataset, test_dataset, user_groups, ntokens


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def add_delta_weights(global_model, average_delta_weights):
    """
    Returns the delta of the previous global model weights and the model after local training.
    """
    w_delta_average = average_delta_weights
    global_m = global_model.state_dict()
    for key in w_delta_average.keys():
        w_delta_average[key] = torch.add(global_m[key], w_delta_average[key])
    return w_delta_average
