import os
import torch
import numpy as np

from user import User


def __img_per_shard(dataset, num_users, shard_per_user=2):
    return int(len(dataset) / num_users / shard_per_user), num_users * shard_per_user


def mnist_iid(dataset, num_users, dataset_name):
    num_items = int(len(dataset) / num_users)
    list_users, all_idxs = [User(i, dataset=dataset_name) for i in range(num_users)], [i for i in range(len(dataset))]
    for i in range(num_users):
        list_users[i].data = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - list_users[i].data)
    return list_users


def mnist_noniid(dataset, num_users, dataset_name):
    num_imgs, num_shards = __img_per_shard(dataset, num_users)
    idx_shard = [i for i in range(num_shards)]
    list_users = [User(i, dataset=dataset_name) for i in range(num_users)]
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            list_users[i].data = np.concatenate(
                (list_users[i].data, idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return list_users


def mnist_noniid_unequal(dataset, num_users, dataset_name):
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    list_users = [User(i, dataset=dataset_name) for i in range(num_users)]
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard = 1
    max_shard = 30

    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                list_users[i].data = np.concatenate(
                    (list_users[i].data, idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        random_shard_size = random_shard_size - 1
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                list_users[i].data = np.concatenate(
                    (list_users[i].data, idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                list_users[i].data = np.concatenate(
                    (list_users[i].data, idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(list_users, key=lambda x: len(list_users[x].data))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            for rand in rand_set:
                k.data = np.concatenate(
                    (k.data, idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return list_users


def cifar_iid(dataset, num_users, dataset_name):
    num_items = int(len(dataset) / num_users)
    list_users, all_idxs = [User(i, dataset=dataset_name) for i in range(num_users)], [i for i in range(len(dataset))]
    for i in range(num_users):
        list_users[i].data = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - list_users[i].data)
    return list_users


def cifar_noniid(dataset, num_users, dataset_name):
    num_imgs, num_shards = __img_per_shard(dataset, num_users)
    idx_shard = [i for i in range(num_shards)]
    list_users = [User(i, dataset=dataset_name) for i in range(num_users)]
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            list_users[i].data = np.concatenate(
                (list_users[i].data, idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return list_users


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def wiki_noniid(dataset, num_users, dataset_name):
    idx_shard = [i for i in range(20)]
    list_users = [User(i, dataset=dataset_name) for i in range(num_users)]
    for i in range(num_users):
        data = dataset[i * 1000:(i + 1) * 1000, :]
        list_users[i].data = data
    return list_users
