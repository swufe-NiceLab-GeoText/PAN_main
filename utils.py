import torch
import numpy as np
import pandas as pd
import collections
from collections import Counter
import argparse
import pickle

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def gen_pin(total_data, flag='train'):
    # 统计POI标签的词频
    counted_data = Counter(flatten(total_data))
    sorted_dict = dict(sorted(counted_data.items(), key=lambda item: item[1], reverse=True))
    np.save("./embeddings/TKY_poi_pin.npy",sorted_dict)
    # with open(f'./embeddings/TKY_poi_pin.pkl', 'wb') as f:
    #     # 使用pickle.dump()将对象写入文件
    #     pickle.dump(sorted_dict, f)
    # sum_data = sum(counted_data.values())
    if flag == 'test':
        data_pin = {item: count for item, count in sorted_dict.items()}  # 作图使用
    else:
        sorted_dict.pop(total_data[0][0])
        data_pin = {item: np.log(count) for item, count in sorted_dict.items()}  # 训练用
        # data_pin = {item: count for item, count in sorted_dict.items()}  # 训练用
    # np.save('./embeddings/{}_poi_pin'.format(city), data_pin)
    pin_max = max(data_pin.values())
    pin_min = min(data_pin.values())
    # pin_mean = np.mean(list(data_pin.values()))
    # pin_std = np.std(list(data_pin.values()))
    if flag == 'train':
        return {key: (value - pin_min) / (pin_max - pin_min) for key, value in data_pin.items()}
        # return {key: (value - pin_mean) / pin_std for key, value in data_pin.items()}
    elif flag == 'test':
        return data_pin
    else:
        return


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str) and not isinstance(el, int):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def calculate_user_acc(predict_pois, user, user_pois):
    acc = 0
    for i in range(len(predict_pois)):
        # a = user_pois[user[i]]
        if predict_pois[i] in user_pois[user[i]]:
            acc += 1
    return acc


def calculate_acc(predict_pois, real_pois):
    acc = 0
    for i in range(len(predict_pois)):
        if predict_pois[i] == real_pois[i]:
            acc += 1
    return acc


def get_max_index(list):
    length = len(list) - 1
    return length


def build_dictionary(train_file, test_file, voc_poi):
    with open(train_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split('\t')[1:]
            for item in items:
                if item not in voc_poi:
                    voc_poi.append(item)
    with open(test_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split('\t')[1:]
            for item in items:
                if item not in voc_poi:
                    voc_poi.append(item)
    # voc_poi.append('start')
    return voc_poi


def extract_words_vocab(voc_poi):
    int_to_vocab = {idx + 1: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])  # 取最大长度
    lengths = [len(sentence) for sentence in sentence_batch]
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch], lengths


# def pad_sentence_batch(sentence_batch, pad_idx):
#     max_sentence = max([len(sentence) for sentence in sentence_batch])  # 取最大长度
#     lengths_list = [len(sentence) for sentence in sentence_batch]
#     return [sentence + [pad_idx] * (max_sentence - len(sentence)) for sentence in sentence_batch], lengths_list


# Read data
def read_data(train_file, test_file):
    Train_DATA = []  # USER-POI
    Train_USER = []  # USERID
    Test_DATA = []
    Test_USER = []
    # T_DATA = {}
    fread_train = open(train_file, 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_DATA.append(data_line)
        Train_USER.append(line[0])
        # T_DATA.setdefault(line[0], []).append(data_line)

    fread_train = open(test_file, 'r')
    for lines in fread_train.readlines():
        line = lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_DATA.append(data_line)
        Test_USER.append(line[0])
    print('Train Size', len(Train_DATA))
    print('total trajectory', len(Test_DATA) + len(Train_DATA))
    # return T_DATA, Train_DATA, Train_USER, Test_DATA, Test_USER
    return Train_DATA, Train_USER, Test_DATA, Test_USER


# poi_time
def load_poi_time(poi_time_file, int_to_vocab):
    poi_time_list = []
    poi_time_graph = pd.read_csv(poi_time_file, index_col=0)
    # print(poi_time_graph
    for poiid in int_to_vocab.keys():
        voc = int_to_vocab[poiid]
        if voc == 'END':
            poi_time_list.append([0.0] * 24)
        else:
            poi_time_list.append(poi_time_graph.loc[eval(voc)].tolist())
        # print(poiid, poi_time_graph.loc[eval(voc)].tolist())
    # [print(i) for i in poi_time_list]
    return poi_time_list


# poi_cat
def load_poi_cat(poi_cat_file, int_to_vocab):
    poi_cat_list = []
    poi_cat_graph = pd.read_csv(poi_cat_file, index_col=0)
    for poiid in int_to_vocab.keys():
        voc = int_to_vocab[poiid]
        poi_cat_list.append(poi_cat_graph.loc[eval(voc)].tolist())
    return poi_cat_list


# convert data
def convert_data(DATA, vocab_to_int):
    new_DATA = list()
    for i in range(len(DATA)):  # TRAIN
        temp = list()
        for j in range(len(DATA[i])):
            temp.append(vocab_to_int[DATA[i][j]])
        new_DATA.append(temp)
    return new_DATA


def convert_het_data(DATA):
    new_DATA = list()
    for i in range(len(DATA)):  # TRAIN
        temp = list()
        for j in range(len(DATA[i])):
            temp.append(int(DATA[i][j]))
        new_DATA.append(temp)
    return new_DATA


# position encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # sin（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # cos；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding = torch.tensor(pos_encoding).float()
    return pos_encoding


def gen_near_poi(lng_lat_file, int_to_vocab, near_dist, save_path, data_raw='foursquare'):
    # 读取经纬度
    raw_file = lng_lat_file
    lng_lat_dict = dict()
    with open(raw_file, 'r') as f:
        if data_raw == 'foursquare':
            next(f)  # 跳过第一行标题
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n')
                    poi = line.split(',')[0]
                    longitude = line.split(',')[2][2:]
                    latitude = line.split(',')[3][:-2]
                    lng_lat = tuple([longitude, latitude])
                    if poi not in lng_lat_dict.keys():
                        lng_lat_dict.update({poi: lng_lat})
        elif data_raw == 'gowalla':
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split('\t')
                    poi, longitude, latitude = line[0], line[1], line[2]
                    lng_lat = tuple([longitude, latitude])
                    lng_lat_dict.update({poi: lng_lat})
        else:
            return
        f.close()

    poi_lng_lat = {}
    for poiid in int_to_vocab.keys():
        voc = int_to_vocab[poiid]
        if poiid not in poi_lng_lat.keys():
            poi_lng_lat.update({poiid: lng_lat_dict[voc]})

    num_poi = len(poi_lng_lat)
    poi_near = {}
    for index in range(1, num_poi + 1):
        temp = []
        for node in range(num_poi - index - 1):
            dist = calc_dist_vec(poi_lng_lat[index][0], poi_lng_lat[index][1],
                                 poi_lng_lat[node + index + 1][0], poi_lng_lat[node + index + 1][1])
            if dist <= near_dist:
                temp.append(node + index + 1)
        poi_near.update({index: temp})

    np.save(save_path, poi_near)
    return poi_near


def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(float(longitudes1))
    lat1 = np.radians(float(latitudes1))
    lng2 = np.radians(float(longitudes2))
    lat2 = np.radians(float(latitudes2))
    radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist = 2 * radius * np.arcsin(np.sqrt(
        (np.sin(0.5 * dlat)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng)) ** 2))
    return dist


class EarlyStopping:
    '''
    使用早停法，快速得到结果
    '''

    def __init__(self, patience=10, model_name='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.model_name = model_name

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), self.model_name)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), self.model_name)
            self.counter = 0
        return self.early_stop


def get_last_percentage(dictionary, percentage):
    # 对字典进行切片
    keys = list(dictionary.keys())
    num_to_keep = int(len(keys) * percentage)
    last_keys = keys[-num_to_keep:]
    result_dict = {key: dictionary[key] for key in last_keys}

    return result_dict