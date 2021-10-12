import os
import math
import copy
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from csv import writer
from update import LocalUpdate, test_inference
from models.models import CNNMnistSmall, VGG, RNNModel
from utils.utils_training import get_dataset, average_weights, add_delta_weights

from user import UserType
import json

with open('config.json') as config_file:
    config = json.load(config_file)


def main(dataset, payload, num_users, frac, run_name):
    result_folder_tree = os.path.join(os.getcwd(), run_name, dataset,
                                      payload, str(num_users), str(int(frac * 100)))

    if not os.path.exists(result_folder_tree):
        os.makedirs(result_folder_tree)
        os.makedirs(os.path.join(result_folder_tree, "models"))
        os.makedirs(os.path.join(result_folder_tree, "payloads"))

    device = 'cuda' if config["gpu"] else 'cpu'
    # load ldpc matrixes
    H, G, enc_length, preamble1, global_model = None, None, None, None, None

    # load datasets
    train_dataset, test_dataset, user_groups, ntokens = get_dataset(dataset, config["iid"], config["unequal"],
                                                                    num_users)
    # BUILD MODEL
    if dataset == 'mnist':
        global_model = CNNMnistSmall()
    elif dataset == 'cifar10':
        global_model = VGG('VGG11')
    elif dataset == "wiki":
        global_model = RNNModel("LSTM", ntokens, 200, 200, 2, 0.2, True)

    error_correction = config["error_correction"]
    stealthiness_level = config["stealthy"]

    if not global_model:
        print('Configuration Error!')
    global_model.to(device)

    # Training
    epoch = 0
    # Injections
    injections = 0
    # Check when we can start the decoding
    payload_alive = False

    # Define the number of sender users
    m_comp = max(int(config["senders"] * num_users), 1)
    sender_users = np.random.choice(range(num_users), m_comp, replace=False)

    for user in user_groups:
        if user.user_id in sender_users:
            user.user_type = UserType.SENDER

    with tqdm(range(config["epochs"])) as bar:
        for _ in bar:
            local_weights, local_losses = [], []
            m = max(int(frac * num_users), 1)
            np.random.seed(random.randint(100, 1000))
            idxs_users = np.random.choice(range(num_users), m, replace=False)

            global_model.train()

            for idx in idxs_users:
                user = user_groups[idx]
                local_model = LocalUpdate(gpu=config["gpu"], dataset=train_dataset, idxs=user.data,
                                          local_bs=config["local_bs"], dataset_name=dataset)

                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch,
                                                     optimizer=config["optimizer"], lr=config["lr"],
                                                     local_ep=config["local_ep"])

                bar.set_postfix({'Loss': {copy.deepcopy(loss)}})

                if config["store_global"] <= epoch < config["injection"]:
                    user.global_model = copy.deepcopy(global_model)
                    user.previous_round = epoch

                if user.user_type == UserType.SENDER and epoch >= config["injection"]:
                    if enc_length is None:
                        user.global_model = copy.deepcopy(global_model)

                    payload_alive = user.extract_payload(copy.deepcopy(global_model),
                                                         payload,
                                                         result_folder_tree,
                                                         enc_length,
                                                         H, G,
                                                         preamble1,
                                                         error_correction)

                    sender_weights, enc_length, H, G, preamble1 = user.inject_payload(copy.deepcopy(w),
                                                                                      device,
                                                                                      payload,
                                                                                      stealthiness_level,
                                                                                      error_correction)

                    local_weights.append(copy.deepcopy(sender_weights))
                    injections += 1
                else:
                    local_weights.append(copy.deepcopy(w.state_dict()))
                    local_losses.append(copy.deepcopy(loss))

            if epoch >= config["injection"] and payload_alive:
                for idx in idxs_users:
                    user = user_groups[idx]
                    user.extract_payload(copy.deepcopy(global_model),
                                         payload,
                                         result_folder_tree,
                                         enc_length,
                                         H, G,
                                         preamble1,
                                         error_correction)

            global_weights_delta = average_weights(local_weights)
            global_weights = add_delta_weights(copy.deepcopy(global_model), global_weights_delta)
            global_model.load_state_dict(global_weights)

            if epoch % 5 == 0:
                if dataset == "wiki":
                    train_loss = test_inference(config["gpu"], copy.deepcopy(global_model), train_dataset, dataset)
                    with open(os.path.join(result_folder_tree, "acc_loss.csv"), 'a+') as fp:
                        writer_object = writer(fp)
                        writer_object.writerow([epoch, train_loss, math.exp(train_loss)])
                        fp.close()
                else:
                    train_acc, train_loss = test_inference(config["gpu"], copy.deepcopy(global_model), train_dataset,
                                                           dataset)
                    test_acc, test_loss = test_inference(config["gpu"], copy.deepcopy(global_model), test_dataset,
                                                         dataset)

                    with open(os.path.join(result_folder_tree, "acc_loss.csv"), 'a+') as fp:
                        writer_object = writer(fp)
                        writer_object.writerow([epoch, train_acc, train_loss, test_acc, test_loss])
                        fp.close()

            rnd_coverage = sum(
                [1 if u.correctly_extracted and not u.user_type == UserType.SENDER else 0 for u in user_groups])
            with open(os.path.join(result_folder_tree, "coverage.csv"), 'a+') as fp:
                writer_object = writer(fp)
                writer_object.writerow([epoch, rnd_coverage])
                fp.close()
            torch.save(global_model.state_dict(),
                       os.path.join(result_folder_tree, "models", f"checkpoint.epoch{epoch}.pt"))
            epoch += 1


if __name__ == '__main__':
    start_time = time.time()
    for p in config["payload"]:
        for n in config["num_users"]:
            for f in config["frac"]:
                main(config["dataset"],
                     p,  # payload
                     n,  # num_users
                     f,  # frac
                     config["run_name"]
                     )
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
