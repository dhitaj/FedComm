from enum import IntEnum

import numpy as np
import torch
import os

from utils.utils_inject import *
from pyldpc import make_ldpc, encode, decode, get_message


class UserType(IntEnum):
    RECEIVER = 0
    SENDER = 1


class User:
    TARGET_LAYERS = {
        "mnist": ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight'],  # CNN
        "wiki": ['rnn.weight_ih_l0', 'rnn.weight_hh_l0'],  # LSTM
        "cifar10": ['features.8.weight'],  # VGG
    }

    def __init__(self, user_id, user_type=UserType.RECEIVER, seed=42, dataset="mnist"):
        self.__user_id = user_id
        self.__user_type = user_type
        self.__seed = seed
        self.__data = np.array([])
        self.__extracted = False
        self.__correctly_extracted = False
        self.__layers_to_inject = User.TARGET_LAYERS[dataset]
        self.__global_model = None
        self.__previous_round = 0

    @property
    def user_id(self):
        return self.__user_id

    @property
    def user_type(self):
        return self.__user_type

    @user_type.setter
    def user_type(self, new_type):
        self.__user_type = new_type

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, new_data):
        self.__data = new_data

    @property
    def global_model(self):
        return self.__global_model

    @global_model.setter
    def global_model(self, new_model):
        self.__global_model = new_model

    @property
    def previous_round(self):
        return self.__previous_round

    @previous_round.setter
    def previous_round(self, new_previous_round):
        self.__previous_round = new_previous_round

    @property
    def correctly_extracted(self):
        return self.__correctly_extracted

    @correctly_extracted.setter
    def correctly_extracted(self, new_correctly_extracted):
        self.__correctly_extracted = new_correctly_extracted

    @property
    def extracted(self):
        return self.__extracted

    @extracted.setter
    def extracted(self, new_extracted):
        self.__extracted = new_extracted

    def inject_payload(self, model, device, filename_ext, stealthiness_level, error_correction=False):
        bit_to_signal_mapping = {
            1: -1,
            0: 1
        }
        H, G, preamble1 = None, None, None

        model_st_dict = model.state_dict()
        models_w = []
        layer_lengths = dict()

        for layer in self.__layers_to_inject:
            x = model_st_dict[layer].detach().cpu().numpy().flatten()
            layer_lengths[layer] = len(x)
            models_w.extend(list(x))

        spreading_code_length = len(models_w)
        message = bits_from_file("payloads/payload.{}".format(filename_ext))
        gradients = np.array(models_w)

        if error_correction:
            if len(message) > 4000:
                k = 3048
            else:
                k = 96
            d_v = 3
            d_c = 6
            n = k * int(d_c / d_v)
            H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
            k = G.shape[1]

            snr1 = 10000000000000000
            c = []
            remaining_bits = len(message) % k
            chunks = int(len(message) / k)

            for ch in range(chunks):
                c.extend(encode(G, message[ch * k:ch * k + k], snr1))

            last_part = []
            last_part.extend(message[chunks * k:])
            last_part.extend([0] * (k - remaining_bits))

            c.extend(encode(G, last_part, snr1))
            preamble1 = np.sign(np.random.uniform(-1, 1, 100))
            b = np.concatenate((preamble1, c))
        else:
            b = [bit_to_signal_mapping[int(bit)] for bit in message]

        if stealthiness_level == "non":
            gamma = np.sqrt(np.var(gradients)) / np.sqrt(len(b))
        elif stealthiness_level == "inter":
            gamma = np.sqrt(np.var(gradients)) / np.sqrt(2 * len(b))
        else:
            gamma = 0.1 * np.sqrt(np.var(gradients)) / np.sqrt(len(b))

        if stealthiness_level == "non":
            models_w = [0.0] * len(models_w)
        elif stealthiness_level == "inter":
            half_stealthy_coeff = 1 / np.sqrt(2)
            models_w = [half_stealthy_coeff * el for el in models_w]

        np.random.seed(self.__seed)
        for i, bit in enumerate(b):
            spreading_code = np.random.choice([-1, 1], size=spreading_code_length)
            current_bit_cdma_signal = gamma * spreading_code * bit
            models_w = np.add(models_w, current_bit_cdma_signal)

        curr_index = 0
        for layer in self.__layers_to_inject:
            x = np.array(models_w[curr_index:curr_index + layer_lengths[layer]])
            model_st_dict[layer] = torch.from_numpy(np.reshape(x, model_st_dict[layer].shape)).to(device)
            curr_index = curr_index + layer_lengths[layer]

        return model_st_dict, len(b), H, G, preamble1

    def extract_payload(self, model, filename_ext, result_folder_tree, enc_length, H, G, preamble1,
                        error_correction=False):

        if self.global_model is None or enc_length is None:
            return False

        extraction_path = os.path.join(result_folder_tree, "payloads",
                                       "{}_ext_payload.{}".format(str(self.user_id), filename_ext))
        st_dict_prev = self.global_model.state_dict()
        st_dict_next = model.state_dict()

        models_w_prev = []
        models_w_curr = []

        layer_lengths = dict()
        total_params = 0

        intended_payload = bits_from_file("payloads/payload.{}".format(filename_ext))

        for layer in self.__layers_to_inject:
            x_prev = st_dict_prev[layer].detach().cpu().numpy().flatten()
            models_w_prev.extend(list(x_prev))
            x_curr = st_dict_next[layer].detach().cpu().numpy().flatten()
            models_w_curr.extend(list(x_curr))
            layer_lengths[layer] = len(x_prev)
            total_params += len(x_prev)

        models_w_prev = np.array(models_w_prev)
        models_w_curr = np.array(models_w_curr)
        models_w_delta = np.subtract(models_w_curr, models_w_prev)
        spreading_code_length = len(models_w_delta)

        x = []
        ys = []
        np.random.seed(self.__seed)
        for i in range(enc_length):
            spreading_code = np.random.choice([-1, 1], size=spreading_code_length)
            y_i = np.matmul(spreading_code.T, models_w_delta)
            ys.append(y_i)
            if not error_correction:
                x.append(0 if y_i > 0 else 1)

        if error_correction:
            y = np.array(ys)
            gain = np.mean(np.multiply(y[:100], preamble1))
            sigma = np.std(np.multiply(y[:100], preamble1) / gain)
            snr = -20 * np.log10(sigma)

            k = G.shape[0]
            y = y[100:]
            chunks = int(len(y) / k)

            for ch in range(chunks):
                d = decode(H, y[ch * k:ch * k + k] / gain, snr)
                x.extend(get_message(G, d))

        bits_to_file(extraction_path, x[:len(intended_payload)])

        if intended_payload == x[:len(intended_payload)] and not self.__user_type == UserType.SENDER:
            self.__correctly_extracted = True

        return intended_payload == x[:len(intended_payload)]
