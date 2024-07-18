import numpy as np
import torch
from torch.utils.data import TensorDataset
from utils import build_dictionary, read_data, flatten, extract_words_vocab, gen_pin, \
    load_poi_time, load_poi_cat
from torch.utils.data import DataLoader
from model.HetEmb import HetEmb
from torch_geometric.data import Data
import pickle


def model_data_loader(args):
    city = args.dataset_city
    train_file = 'data/' + city + '/' + city + '_traj_train.txt'
    test_file = 'data/' + city + '/' + city + '_traj_test.txt'
    time_train_file = 'data/' + city + '/' + city + '_traj_time_train.txt'
    time_test_file = 'data/' + city + '/' + city + '_traj_time_test.txt'
    cat_train_file = 'data/' + city + '/' + city + '_traj_cat_train.txt'
    cat_test_file = 'data/' + city + '/' + city + '_traj_cat_test.txt'
    poi_time_file = 'data/' + city + '/' + city + '_poi_time.txt'
    poi_cat_file = 'data/' + city + '/' + city + '_poi_cat.txt'

    train_traj, train_time, train_cat, test_traj, test_time, test_cat, \
        history_traj, history_time, history_cat, train_user, test_user, int_to_vocab, vocab_to_int = data_process(
        train_file, test_file, time_train_file, time_test_file, cat_train_file, cat_test_file)

    # traj_graph = get_graph(train_traj+test_traj, int_to_vocab)

    total_user = set(map(int, train_user + test_user))
    user_num = max(total_user) + 1
    poi_num = len(int_to_vocab) + 1 + 1


    train_data_set = MyDataset(user=train_user, data=train_traj, time=train_time, cat=train_cat,
                               his_poi=history_traj, his_time=history_time, his_cat=history_cat,
                               poi_num=poi_num, padding_idx=0)

    test_data_set = MyDataset(user=test_user, data=test_traj, time=test_time, cat=test_cat,
                              his_poi=history_traj, his_time=history_time, his_cat=history_cat,
                              poi_num=poi_num, padding_idx=0)

    test_poi_pin_label = gen_pin(test_data_set.label.tolist(), 'test')
    train_poi_pin = gen_pin([row[1:] for row in train_traj], 'test')
    test_niche_label_poi = {k: v for k, v in train_poi_pin.items()
                            if k in test_poi_pin_label.keys() and v < 30}
    # with open(f'./results/{city}_niche_poi_label.pkl', 'wb') as f:
    #     # 使用pickle.dump()将对象写入文件
    #     pickle.dump(test_niche_label_poi, f)

    niche_test_traj = []
    niche_test_time = []
    niche_test_cat = []
    niche_test_user = []
    for user, traj, time, cat in zip(test_user, test_traj, test_time, test_cat):
        if traj[-1] in test_niche_label_poi.keys():
            niche_test_traj.append(traj)
            niche_test_time.append(time)
            niche_test_cat.append(cat)
            niche_test_user.append(user)
        else:
            continue
    niche_data_set = MyDataset(user=niche_test_user, data=niche_test_traj, time=niche_test_time, cat=niche_test_cat,
                               his_poi=history_traj, his_time=history_time, his_cat=history_cat,
                               poi_num=poi_num, padding_idx=0)

    # with open(f'./results/{city}_niche_dataset.pkl', 'wb') as f:
    #     # 使用pickle.dump()将对象写入文件
    #     pickle.dump(niche_data_set, f)

    train_data_iter = DataLoader(dataset=train_data_set, batch_size=args.batch_size, shuffle=True)
    test_data_iter = DataLoader(dataset=test_data_set, batch_size=args.batch_size)
    niche_data_iter = DataLoader(dataset=niche_data_set, batch_size=args.batch_size)

    cat_data = train_cat + test_cat
    time_data = train_time + test_time
    poi_data = train_traj + test_traj

    ct_dict, cp_onehot = cat2poi(cat_data, poi_data, args.cat_num, poi_num)
    tp_dict, tp_onehot = cat2poi(time_data, poi_data, args.time_num, poi_num)
    # cp_onehot = get_ct_one_hot(args.cat_num, poi_num, ct_dict)
    # tp_onehot = get_ct_one_hot(args.time_num, poi_num, tp_dict)

    model_kwargs = {}
    model_kwargs.update({'tp_one_hot': tp_onehot.to(args.device)})
    model_kwargs.update({'ct_one_hot': cp_onehot.to(args.device)})

    # poi_pin = gen_pin(train_data_set.label.tolist())
    poi_pin = gen_pin(poi_data)
    poi_freq = torch.zeros(poi_num, )
    for key, value in poi_pin.items():
        poi_freq[key] = value

    model_kwargs.update({'poi_freq': poi_freq.to(args.device)})
    model_kwargs.update({'niche_label': test_niche_label_poi})
    model_kwargs.update({'poi_num': poi_num})
    model_kwargs.update({'user_num': user_num})

    return train_data_iter, test_data_iter, niche_data_iter, model_kwargs


def get_ct_one_hot(cat_num, poi_num, ct_dict):
    ct_one_hot = torch.zeros(cat_num + 1, poi_num)
    for i in range(len(ct_dict)):
        if i in ct_dict.keys():
            time_poi = ct_dict[i]
            ct_one_hot[i, time_poi] = 1
        else:
            continue
    return ct_one_hot


def get_graph(trajectories, int_to_vocab):
    # 构建图数据
    edges = []
    nodes = []
    # 添加空缺占位符
    nodes.append(0)

    # 邻接矩阵
    for traj in trajectories:
        for i in range(len(traj) - 1):
            edges.append((traj[i], traj[i + 1]))
        for poi in traj:
            if poi not in nodes:
                nodes.append(poi)

    n = len(int_to_vocab) + 2
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for edge in edges:
        u, v = edge
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1  # 因为是无向图
    adj_matrix = np.array(adj_matrix)

    # 添加起始符号
    nodes.append(n)

    # 创建图数据对象
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node = torch.tensor(nodes, dtype=torch.long)
    adj_label = torch.tensor(adj_matrix, dtype=torch.float)
    # adj_label = torch.ones_like(edge_index, dtype=torch.float)[0]
    # x = torch.eye(num_nodes, dtype=torch.float)  # 节点特征矩阵，使用单位矩阵

    graph_data = Data(x=node, edge_index=edge_index, adj_label=adj_label)


def cat2poi(cat_data, poi_data, cat_num, poi_num):
    ct_dict = {}
    ct_one_hot = torch.zeros(cat_num + 2, poi_num)
    # cat_data = set(cat_data)
    for i in range(len(cat_data)):
        for j in range(len(cat_data[i])):
            ct_dict.setdefault(cat_data[i][j], []).append(poi_data[i][j])
            ct_one_hot[cat_data[i][j], poi_data[i][j]] += 1
    new_dict = {k: list(set(v)) for k, v in ct_dict.items()}
    return new_dict, ct_one_hot


def user_active_divide(his_traj, his_cat, his_time, divide_start, divide_end):
    if divide_start < 0 or divide_end > 1:
        print("divide_start or divide_end is wrong")
    else:
        user_traj_count = {key: len(value) for key, value in his_traj.items()}

        user_traj_count = list(sorted(user_traj_count.items(), key=lambda x: x[1], reverse=True))

        divide_user_count = user_traj_count[
                            int(len(user_traj_count) * divide_start):int(len(user_traj_count) * divide_end)]
        divide_user = flatten([[x[0]] * x[1] for x in divide_user_count])
        divide_user_only = [x[0] for x in divide_user_count]
        divide_traj = [traj for user in divide_user_only for traj in his_traj[user]]
        divide_cat = [cat for user in divide_user_only for cat in his_cat[user]]
        divide_time = [time for user in divide_user_only for time in his_time[user]]
        return divide_traj, divide_cat, divide_time, divide_user


def user_random_divide(his_traj, his_cat, his_time, t_his_traj, t_his_cat, t_his_time, scale):
    if scale < 0 or scale > 1:
        print("scale is wrong")
    else:
        user_list = [key for key in his_traj.keys()]
        user_count1 = {user: len(value) for user, value in his_traj.items()}
        user_count2 = {user: len(value) for user, value in t_his_traj.items()}

        np.random.shuffle(user_list)
        old_user = user_list[:int(len(user_list) * scale)]
        new_user = user_list[int(len(user_list) * scale):-1]
        train_user = flatten([[user_id] * user_count1[user_id] for user_id in old_user])
        test_user = flatten([[t_user] * user_count2[t_user] for t_user in new_user])
        train_traj = [traj for user in old_user for traj in his_traj[user]]
        train_cat = [cat for user in old_user for cat in his_cat[user]]
        train_time = [time for user in old_user for time in his_time[user]]

        test_traj = [traj for user in new_user for traj in t_his_traj[user]]
        test_cat = [cat for user in new_user for cat in t_his_cat[user]]
        test_time = [time for user in new_user for time in t_his_time[user]]
    return train_traj, train_cat, train_time, test_traj, test_cat, test_time, train_user, test_user


def data_process(traj_train, traj_test, time_train, time_test, cat_train, cat_test):
    voc_poi = []
    voc_poi = build_dictionary(traj_train, traj_test, voc_poi)
    Train_DATA, Train_USER, Test_DATA, Test_USER = read_data(traj_train, traj_test)
    Train_TIME, _, Test_TIME, _ = read_data(time_train, time_test)
    Train_CAT, _, Test_CAT, _ = read_data(cat_train, cat_test)

    int_to_vocab, vocab_to_int = extract_words_vocab(voc_poi)

    train_traj = convert_data(Train_DATA, vocab_to_int)
    test_traj = convert_data(Test_DATA, vocab_to_int)
    train_time = convert_het_data(Train_TIME)
    test_time = convert_het_data(Test_TIME)
    train_cat = convert_het_data(Train_CAT)
    test_cat = convert_het_data(Test_CAT)

    # train_time = time_sort(train_time)
    # test_time = time_sort(test_time)
    his_traj, his_time, his_cat = his_data_gen(train_traj, train_time, train_cat, Train_USER)
    t_his_traj, t_his_time, t_his_cat = his_data_gen(test_traj, test_time, test_cat, Test_USER)

    # Divided the dataset by activity
    # train_traj, train_cat, train_time, Train_USER = user_active_divide(his_traj, his_cat, his_time, 0, 0.7)
    # test_traj, test_cat, test_time, Test_USER = user_active_divide(t_his_traj, t_his_cat, t_his_time, 0.7, 1)

    # train_traj, train_cat, train_time, test_traj, test_cat, test_time, Train_USER, Test_USER = \
    #               user_divide(his_traj, his_cat, his_time, t_his_traj, t_his_cat, t_his_time, 0.7)



    his_traj = read_history_data(his_traj)
    his_time = read_history_data(his_time)
    his_cat = read_history_data(his_cat)

    train_traj, test_traj = add_sos(train_traj, test_traj)
    train_time, test_time = add_sos(train_time, test_time)
    train_cat, test_cat = add_sos(train_cat, test_cat)

    T = train_traj + test_traj
    total_check = len(flatten(T))
    total_user = set(flatten(Train_USER + Test_USER))
    user_number = len(set(total_user))

    print(len(int_to_vocab))
    print('Dictionary Length', len(int_to_vocab), 'POI number', len(int_to_vocab) - 3)
    TOTAL_POI = len(int_to_vocab)
    print('Total check-ins', total_check, TOTAL_POI)
    print('Total Users', user_number)

    return train_traj, train_time, train_cat, test_traj, test_time, test_cat, his_traj, his_time, his_cat, \
        Train_USER, Test_USER, int_to_vocab, vocab_to_int


def add_sos(train_data, test_data):
    sos = 0
    for list in train_data + test_data:
        temp = max(list)
        if temp > sos:
            sos = temp
    sos = sos + 1
    for i in range(len(train_data)):
        train_data[i] = [sos] + train_data[i]

    for i in range(len(test_data)):
        test_data[i] = [sos] + test_data[i]
    return train_data, test_data


def read_history_data(data):
    History = {}
    for key in data.keys():  # index char
        temp = data[key]
        History[key] = flatten(temp)
    return History


def his_data_gen(traj, time, cat, user):
    his_traj = {}
    his_time = {}
    his_cat = {}
    for i in range(len(traj)):
        his_traj.setdefault(int(user[i]), []).append(traj[i])
        his_time.setdefault(int(user[i]), []).append(time[i])
        his_cat.setdefault(int(user[i]), []).append(cat[i])
    return his_traj, his_time, his_cat


def time_sort(time):
    for i in range(len(time)):
        if time[i][0] == 999:
            continue
        if time[i][0] > 24:
            temp_time = time[i][0] - 24
        else:
            temp_time = time[i][0]
        for j in range(len(time[i])):
            time[i][j] = time[i][j] - temp_time
            if time[i][j] < 0:
                time[i][j] += 24
            elif time[i][j] == 0:
                if j == len(time[i]) - 1:
                    time[i][j] = 24
                else:
                    time[i][j] = 0

    for i in range(len(time)):
        for j in range(len(time[i])):
            time[i][j] += 1
    return time


def convert_data(data, vocab_to_int):
    new_data = [[vocab_to_int[item] for item in sublist] for sublist in data]
    return new_data


def convert_het_data(data):
    new_data = [[int(item) + 1 for item in sublist] for sublist in data]
    return new_data


class MyDataset(TensorDataset):
    def __init__(self, user, data, time, cat, his_poi, his_time, his_cat, padding_idx, poi_num, use_sos_eos=False):
        super(TensorDataset, self).__init__()

        self.data = data
        self.user = user
        self.time = time
        self.cat = cat
        self.his_poi = his_poi
        self.his_time = his_time
        self.his_cat = his_cat

        self.padding_idx = padding_idx
        self.sos_eos = use_sos_eos
        self.poi_num = poi_num
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        label, data = self.get_label(data)
        label_c, cat = self.get_label(cat)
        label_t, time = self.get_label(time)

        max_poi_id = max(max(row) for row in data)
        sos = [max_poi_id + 1]
        eos = [max_poi_id + 2]

        if self.sos_eos:
            data = self.add_sos_eos(data, sos, eos)
            cat = self.add_sos_eos(cat, sos, eos)
            time = self.add_sos_eos(time, sos, eos)

        user = list(map(int, user))
        data, lengths = self.pad_sentence_batch(data, self.padding_idx)
        time, _ = self.pad_sentence_batch(time, self.padding_idx)
        cat, _ = self.pad_sentence_batch(cat, self.padding_idx)
        self.his_poi, self.his_time, self.his_cat, self.his_length = self.pad_history(user, his_poi, his_time, his_cat)

        # 将数据进行转换
        self.data = torch.tensor(data).to(self.device)
        self.time = torch.tensor(time).to(self.device)
        self.cat = torch.tensor(cat).to(self.device)

        self.label = torch.tensor(label).to(self.device)
        self.label_c = torch.tensor(label_c).to(self.device)
        self.label_t = torch.tensor(label_t).to(self.device)

        self.user = torch.tensor(user).to(self.device)

        self.lengths = torch.tensor(lengths)

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, index):
        return \
            self.user[index], self.label[index], self.label_t[index], self.label_c[index], \
                self.data[index], self.time[index], self.cat[index], self.lengths[index], \
                self.his_poi[index], self.his_time[index], self.his_cat[index], self.his_length[index]

    def get_poi_num(self):
        return self.poi_num

    def pad_sentence_batch(self, sentence_batch, pad_idx):
        max_sentence_length = max([len(sentence) for sentence in sentence_batch])  # 取最大长度
        pad_sentences = [sentence + [pad_idx] * (max_sentence_length - len(sentence)) for sentence in sentence_batch]
        lengths_list = [len(sentence) for sentence in sentence_batch]
        return pad_sentences, lengths_list

    def add_sos_eos(self, trajectoreis, sos, eos):
        for i in range(len(trajectoreis)):
            trajectoreis[i] = sos + trajectoreis[i]
        return trajectoreis

    def get_label(self, trajectories):
        labels = [trajectory[-1] for trajectory in trajectories]
        trajectories_without_labels = [trajectory[:-1] for trajectory in trajectories]
        return labels, trajectories_without_labels

    def pad_history(self, user, history_traj, history_time, history_cat):
        his_traj = [history_traj[uid] for uid in user]
        his_time = [history_time[uid] for uid in user]
        his_cat = [history_cat[uid] for uid in user]

        his_traj, his_length = self.pad_sentence_batch(his_traj, 0)
        his_time, _ = self.pad_sentence_batch(his_time, 0)
        his_cat, _ = self.pad_sentence_batch(his_cat, 0)

        his_traj = torch.tensor(his_traj).to(self.device)
        his_time = torch.tensor(his_time).to(self.device)
        his_cat = torch.tensor(his_cat).to(self.device)
        his_length = torch.tensor(his_length)
        return his_traj, his_time, his_cat, his_length
