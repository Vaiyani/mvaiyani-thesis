from data_flow import data_provider
from exp.exp_basic import Exp_Basic
from models.Transformer import Transformer
from models.Autoformer import Autoformer
from models.Informer import Informer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main():
    def __init__(self, args):
        self.args = args
        self.train_time = 0
        self.epoc = 0
        self.test_time = 0
        # self.device = torch.device('cuda:1')
        self.device = torch.device("cuda")  ## specify the GPU id's, GPU id's start from 0.
        self.model = nn.DataParallel(self._build_model()).to(self.device)

        # model = nn.DataParallel(model, device_ids=[1, 3])
        # model.to(device)
        # self.model = self._build_model().to(self.device)

    def _build_model(self):
        if self.args.model == 'Transformer':
            model = Transformer(self.args).float()
        elif self.args.model == 'Autoformer':
            model = Autoformer(self.args).float()
        elif self.args.model == 'Informer':
            model = Informer(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)


        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        train_start_time = time.time()
        plt_train_loss = []
        plt_vali_loss = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    # print(outputs.shape,batch_y.shape)
                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())


                loss.backward()
                model_optim.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            plt_train_loss.append(train_loss)
            plt_vali_loss.append(vali_loss)
            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.train_time = time.time() - train_start_time
                self.epoc = epoch
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        self.train_time = time.time() - train_start_time
        self.epoc = epoch
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # plt.plot(plt_vali_loss, label='validation loss')
        # plt.plot(plt_train_loss, label='train loss')
        # plt.legend()
        # plt.savefig('fig/'+ self.args.model +'-loss-' + 'lookback-' + str(self.args.seq_len) + 'future-' + str(self.args.pred_len) + '.png')
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        inputx = []
        self.model.eval()
        test_start_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
        self.test_time = time.time() - test_start_time

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])


        mae, mse, rmse, mape, mspe, rse, corr, r_square = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, R2:{}, train_time:{}, test_time:{}, epoc:{}' .\
              format(mse, mae, rmse, mape, mspe, rse, r_square, self.train_time, self.test_time, self.epoc))
        f = open("understanding_seq_len.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}, rse:{}, R2:{}, train_time:{}, test_time:{}, epoc:{}'.\
              format(mse, mae, rmse, mape, mspe, rse, r_square, self.train_time, self.test_time, self.epoc))
        f.write('\n')
        f.write('\n')
        f.close()
        # plt.clf()
        # plt.plot(preds[-1,:,:].flatten(), label='prediction')
        # plt.plot(trues[-1,:,:].flatten(), label='true')
        # plt.plot(np.array(list(trues[-self.args.pred_len-1, :, :].flatten()) + list(preds[-1, :, :].flatten())), label='prediction')
        # plt.plot(np.array(list(trues[-self.args.pred_len-1, :, :].flatten()) + list(trues[-1, :, :].flatten())), label='true')

        # plt.legend()
        # plt.savefig('fig/'+ self.args.model +'-pred-' + 'lookback-' + str(self.args.seq_len) + 'future-' + str(self.args.pred_len) + '.png')

        return

