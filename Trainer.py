import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metric

from tqdm import tqdm
from utils import EarlyStopping
import pickle


class Trainer(object):
    def __init__(self, args, model, model_kwargs=None):
        self.args = args
        self.model = model
        self.model_kwargs = model_kwargs
        self.poi_num = model_kwargs.get('poi_num')
        # self.niche_flag = 'PAN'
        # self.niche_flag = 'SSDL'
        # self.niche_flag = 'CFPRec'
        self.niche_flag = 'test'
        # self.niche_flag = 'niche'
        if not os.path.isdir('trains'):
            os.mkdir('trains')
        self.model_name = 'trains/{}_PAN_MODEL_checkpoint'.format(args.dataset_city) + '_' \
                          + str(args.embed_size) + '_' \
                          + str(args.hidden_size) + '_' \
                          + str(self.niche_flag) + '.pt'

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps,
                                          weight_decay=args.weight_decay, amsgrad=args.amsgrad)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.stopper = EarlyStopping(patience=args.patience, model_name=self.model_name)

        self.poi_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.cat_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def model_run(self, train_data_iter, test_data_iter, niche_data_iter):
        self.title()
        for epoch in tqdm(range(self.args.epochs)):
            pass
            # self.model_train(train_iters=train_data_iter, epoch=epoch)
            # if (epoch + 1) % 3 == 0:
            #     ACC_1, pre_true = self.model_test(model=self.model, test_iters=test_data_iter, epoch=epoch)
            #     if self.stopper.step(ACC_1, self.model):
            #         break
        self.model.load_state_dict(torch.load(self.model_name))

        ACC_1, _ = self.model_test(model=self.model, test_iters=test_data_iter, epoch=epoch)
        niche_ACC_1, pred_true = self.model_test(model=self.model, test_iters=niche_data_iter, epoch=epoch)

        niche_label_freq = self.model_kwargs.get("niche_label")
        freq_label = {}
        for a in pred_true:
            if a[1] in niche_label_freq.keys():
                poi_freq = niche_label_freq[a[1]]
                if poi_freq not in freq_label:
                    freq_label[poi_freq] = [[a[1], a[2], poi_freq]]
                else:
                    freq_label[poi_freq].append([a[1], a[2], poi_freq])
        with open(f'./results/{self.args.dataset_city}_niche_label_{self.niche_flag}.pkl', 'wb') as f:
            # 使用pickle.dump()将对象写入文件
            pickle.dump(freq_label, f)

    print('Finished Train')

    def model_train(self, train_iters, epoch=None):
        self.model.train()
        epoch_loss = []
        train_bat_acc = 0
        train_bat_cat = 0
        train_total = 0
        for step, (user, label, label_t, label_c, traj, time, cat, length,
                   his_traj, his_time, his_cat, his_length) in enumerate(train_iters):
            self.optimizer.zero_grad()
            logit_poi, logit_cat, _, _ = self.model(traj, time, cat, length, user, his_traj, his_time, his_cat,
                                                    his_length)

            poi_loss = self.poi_loss(logit_poi, label)
            cat_loss = self.cat_loss(logit_cat, label_c)

            pred_poi = torch.argmax(F.softmax(logit_poi, dim=-1), dim=-1)
            pred_cat = torch.argmax(F.softmax(logit_cat, dim=-1), dim=-1)

            loss_all = poi_loss + cat_loss
            # loss_all = poi_loss  # categorical_loss ablation
            #
            loss_all.backward()  # retain_graph=True
            epoch_loss.append(loss_all.item())
            self.optimizer.step()

            train_bat_cat += torch.sum(pred_cat == label_c).item()
            train_bat_acc += torch.sum(pred_poi == label).item()
            train_total += len(label)
        train_acc = train_bat_acc / train_total
        train_cat_acc = train_bat_cat / train_total
        print('\nepoch', epoch + 1, '---train_acc', train_acc,
              '---train_cat_acc', train_cat_acc,
              'loss：', sum(epoch_loss) / len(epoch_loss))
        self.scheduler.step()


    def model_test(self, model, test_iters, epoch):
        correct_per_point = []
        pred_true = []
        test_loss_epoch = []
        attn_weight_list = []
        Test_Acc_1 = 0
        Test_Acc_5 = 0
        Test_Acc_10 = 0
        Test_poi_total = 0
        Y = []
        Prob_Y = []
        test_cat_acc = 0
        test_cat_acc10 = 0
        test_a = 0
        model.eval()
        with torch.no_grad():
            for step, (user, label, label_t, label_c, traj, time, cat, length,
                       his_traj, his_time, his_cat, his_length) in enumerate(test_iters):
                logit_poi, logit_cat, attn_weight, pred_cat10 = model(traj, time, cat, length, user,
                                                                      his_traj, his_time, his_cat, his_length)
                poi_loss = self.poi_loss(logit_poi, label)
                cat_loss = self.cat_loss(logit_cat, label_c)

                attn_weight_list.append(attn_weight)
                # 计算类别准确率：
                pred_cat = torch.argmax(F.softmax(logit_cat, dim=-1), dim=-1)
                test_cat_acc += torch.sum(pred_cat == label_c).item()
                for i in range(label_c.size(0)):
                    if label_c[i] in pred_cat10[i, :]:
                        test_cat_acc10 += 1

                test_loss = poi_loss + cat_loss

                pred_poi = F.softmax(logit_poi, dim=-1)
                predictions = torch.argmax(pred_poi, dim=-1)

                # 保存个案分析
                # for pred_p, true_p, pred_c, true_c in zip(predictions, label, pred_cat, label_c):
                #     flag_p = False
                #     flag_c = False
                #     if pred_p == true_p:
                #         flag_p = True
                #         correct_per_point.append(pred_p.item())
                #     if pred_c == true_c:
                #         flag_c = True
                #     pred_true.append(
                #         tuple([pred_p.item(), true_p.item(), flag_p, pred_c.item(), true_c.item(), flag_c]))

                test_loss_epoch.append(test_loss.item())
                pred_poi = pred_poi.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                B = traj.size(0)
                Test_poi_total += B
                for i in range(B):
                    value = pred_poi[i]
                    tag = label[i]
                    true_value = self.get_onehot(tag)
                    top1 = np.argpartition(a=-value, kth=1)[:1]
                    top5 = np.argpartition(a=-value, kth=5)[:5]
                    top10 = np.argpartition(a=-value, kth=10)[:10]
                    if top1[0] == tag:
                        pred_true.append([top1[0], tag, True])
                        Test_Acc_1 += 1
                    else:
                        pred_true.append([top1[0], tag, False])
                    if tag in top5:
                        Test_Acc_5 += 1
                    if tag in top10:
                        Test_Acc_10 += 1
                    Y.append(true_value)
                    Prob_Y.append(value)

            ACC_1 = Test_Acc_1 / Test_poi_total
            ACC_5 = Test_Acc_5 / Test_poi_total
            ACC_10 = Test_Acc_10 / Test_poi_total
            cat_acc1 = test_cat_acc / Test_poi_total
            cat_acc10 = test_cat_acc10 / Test_poi_total

            print('epoch', epoch + 1,
                  '\nTest accuracy@1：', ACC_1,
                  '\nTest accuracy@5：', ACC_5,
                  '\nTest accuracy@10：', ACC_10,
                  '\nTest loss', sum(test_loss_epoch) / len(test_loss_epoch),
                  '\nTest CAT ACC@1:', cat_acc1,
                  '\nTest CAT ACC@10:', cat_acc10,
                  )

            auc = 0
            Map = 0
            Y = np.array(Y)
            Prob_Y = np.array(Prob_Y)
            auc = metric.roc_auc_score(Y.T, Prob_Y.T, average='micro')
            Map = metric.average_precision_score(Y.T, Prob_Y.T, average='micro')
            print('---------------------------')
            print('AUC_value', auc)  # ,,average='micro'
            print('MAP_value', Map)  # ,,average='micro'
            self.output(epoch + 1, ACC_1, ACC_5, ACC_10, auc, Map)
            return ACC_1, pred_true

    def get_onehot(self, index):
        x = [0] * self.poi_num
        x[index] = 1
        return x

    def output(self, index, ACC_1, ACC_5, ACC_10, auc, map):
        fw_res = open('./results/PAN_Result_{}.txt'.format(self.args.dataset_city), 'a')
        fw_res.flush()
        fw_res.write(
            'epoch' + str(index) +
            '\tACC@1:\t{:.4f}'.format(ACC_1) + '\tACC@5:\t{:.4f}'.format(ACC_5) + '\tACC@10:\t{:.4f}'.format(ACC_10) +
            '\tAUC:\t{:.4f}'.format(auc) + '\tMAP:\t{:.4f}'.format(map) + '\n')
        fw_res.close()

    def title(self):
        if not os.path.isdir('results'):
            os.mkdir('results')
        fw_res = open('./results/PAN_Result_{}.txt'.format(self.args.dataset_city), 'a')
        fw_res.flush()
        fw_res.write(
            '-----------this is title---------'
            '\n{}_PAN_MODEL'.format(self.args.dataset_city) + '_' + str(self.args.embed_size) + '_'
            + str(self.args.hidden_size) + '\n')
        fw_res.close()
