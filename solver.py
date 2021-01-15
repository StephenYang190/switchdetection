import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import time
from mobilenetv3 import MobileNetV3_Small
from tqdm import tqdm


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.device = torch.device(config.device_id) if torch.cuda.is_available() else torch.device("cpu")
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=config.device_id))
            self.net.eval()
        if config.mode == 'train':
            self.net.train()


    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = MobileNetV3_Small(2)

        if self.config.cuda:
            self.net.to(self.device)

        if self.config.pre_model is not None:
            self.net.load_state_dict(torch.load(self.config.pre_model, map_location=self.config.device_id))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'MobileNetV3')

    def test(self):
        print('Loading pre-trained model from %s...' % self.config.model)
        self.net.load_state_dict(torch.load(self.config.model))
        self.net.eval()
        s = nn.Sigmoid()

        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name = data_batch['image'], data_batch['name'][0]
            print("Proccess :" + name)

            with torch.no_grad():
                images = images.to(self.device)

                preds = self.net(images)
                preds = s(preds)
                _, l = torch.max(preds, 1)
                if l == 0:
                    print('close')
                else:
                    print('open')

        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        self.optimizer.zero_grad()
        MinLoss = 100
        for epoch in range(self.config.epoch):
            TotalLoss = 0
            ValLoss = 0
            time_s = time.time()
            print("epoch: %2d/%2d || " % (epoch, self.config.epoch), end='')
            itertqdm = tqdm(self.train_loader, desc='%d / %d' % (epoch, self.config.epoch))
            for i, data_batch in enumerate(itertqdm):
                img_in, target = data_batch['data_image'].to(self.device), data_batch['data_label'].to(self.device)
                output = self.net(img_in)

                loss_fun = nn.BCEWithLogitsLoss()
                Loss = loss_fun(output, target)
                if i < 57:
                    TotalLoss += Loss
                    Loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                else:
                    ValLoss += Loss
                    Loss.backward()
                    self.optimizer.zero_grad()

                itertqdm.set_postfix(Totalloss=TotalLoss, Valloss=ValLoss)

            time_e = time.time()
            print(' || Loss : %10.4f || Time : %f s' % (TotalLoss / iter_num, time_e - time_s))
            if epoch == 50:
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0005, weight_decay=self.wd)
            elif epoch == 100:
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=self.wd)
            if ValLoss < MinLoss:
                MinLoss = ValLoss
                torch.save(self.net.state_dict(), '%s/best_result.pth' % self.config.save_folder)

        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)