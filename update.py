import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(35, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data.to('cuda'), target.to('cuda')


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, gpu, dataset, idxs, local_bs, dataset_name):
        if dataset_name != 'wiki':
            self.trainloader, self.validloader, self.testloader = self.train_val_test(local_bs, dataset, list(idxs))

        self.device = 'cuda' if gpu else 'cpu'
        self.dataset_name = dataset_name
        if dataset_name == 'wiki':
            self.criterion = nn.NLLLoss().to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.idxs = idxs
        self.local_bs = local_bs
        self.prev_model = None
        self.last_round = None
        self.layer_changes = dict()

    def train_val_test(self, local_bs, dataset, idxs):
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val) / 1), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test) / 1), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, optimizer, lr, local_ep):
        self.prev_model = copy.deepcopy(model.state_dict())
        self.last_round = global_round

        model.train()
        epoch_loss = []

        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.5)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=1e-4)

        for _ in range(local_ep):
            batch_loss = []
            if self.dataset_name == 'wiki':
                hidden = model.init_hidden(20)

                for batch, i in enumerate(range(0, self.idxs.size(0) - 1, 35)):
                    data, targets = get_batch(self.idxs, i)
                    model.zero_grad()

                    hidden = repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                    loss = self.criterion(output, targets)
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    for p in model.parameters():
                        p.data.add_(p.grad, alpha=-20)

                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
                for _, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        delta_model = self.parameter_delta_weights(copy.deepcopy(model.state_dict()))
        model.load_state_dict(delta_model)
        return model, sum(epoch_loss) / len(epoch_loss)

    def parameter_delta_weights(self, latest_local_model):
        w_delta = latest_local_model
        for key in w_delta.keys():
            w_delta[key] = torch.subtract(w_delta[key], self.prev_model[key])
        return w_delta

    @torch.no_grad()
    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


@torch.no_grad()
def test_inference(gpu, model, test_dataset, dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if gpu else 'cpu'

    if dataset == "wiki":
        criterion = torch.nn.NLLLoss().to(device)
        hidden = model.init_hidden(20)
        total_loss = 0.
        for batch, i in enumerate(range(0, test_dataset.size(0) - 1, 35)):
            data, targets = get_batch(test_dataset, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)

            total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(test_dataset) - 1)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss / len(testloader)
