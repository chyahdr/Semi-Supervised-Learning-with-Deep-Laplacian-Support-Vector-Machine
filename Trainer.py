#!coding:utf-8
import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import torch.nn as nn
from pathlib import Path
from util.datasets import NO_LABEL
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE # 导入tSNE类
import numpy as np
class PseudoLabel:

    def __init__(self, model, optimizer, loss_fn, device, config, writer=None, save_dir=None, save_freq=5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.device = device
        self.writer = writer
        self.labeled_bs = config.labeled_batch_size
        self.global_step = 0
        self.epoch = 0
        self.T1, self.T2 = config.t1, config.t2
        self.af = config.af
        
    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_loss = []
        accuracy = []
        labeled_n = 0
        torch.autograd.set_detect_anomaly(True)
        mode = "train" if is_train else "test"
        for batch_idx, (data, targets) in enumerate(data_loader):
            self.global_step += batch_idx
            data, targets = data.to(self.device), targets.to(self.device)
            data3 = data.reshape(data.shape[0], -1)
            dist=torch.cdist(data3, data3, p=2)
            L=creatLap(dist)
            outputs, weight_fc1, outsize, Dis, out = self.model(data)
            # temp = torch.ones((1, Dis.size(0))) * float('inf')
            # temp = temp.to(self.device)
            # Dis = Dis + torch.diag(temp[0])

            if is_train:
                # Dis2 = torch.cdist(data, data, p=2)
                # Y,I = torch.sort(Dis, dim=1)
                # W = torch.zeros_like(Dis)
                # for ii in range(Dis.size(0)):
                #     W[ii, I[ii, :30]] = 1
                # B = W + W.t()
                # W = Dis.clone()
                # mask = B < 0.9
                # W = W * torch.logical_not(mask)
                # t = 10
                # W = torch.exp(-W.pow(2) / (2 * t * t))
                # W_exp = W * torch.logical_not(mask)
                # D = torch.sum(W_exp, dim=1)
                # L = torch.diag(D) - W_exp

                first_row = weight_fc1[0, :]
                squared_elements1 = torch.square(first_row)
                sum_of_squares1 = torch.sum(squared_elements1)
                #square_root_of_sum1 = torch.sqrt(sum_of_squares1)
                labeled_bs = self.labeled_bs
                targets1 = targets.clone()
                targets1[targets1 == 0] = -1
                outputs = torch.squeeze(outputs)
                # outputs1=torch.sign(outputs[:labeled_bs])
                labeled_loss =torch.sum(torch.clamp(1-torch.mul(outputs, targets1),min=0)[:labeled_bs])/ labeled_bs
                # labeled_loss = torch.sum(torch.abs(1 - torch.mul(outputs1, targets1[:labeled_bs]))) / labeled_bs
                outputs = torch.unsqueeze(outputs,1)
                unlabeled_loss = outputs.t().mm(L).mm(outputs)/ (data.size(0)+1e-10)
                loss=labeled_loss +0.01*unlabeled_loss+0.1*sum_of_squares1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loop_loss.append(loss.item() / len(data_loader))
                outputs = torch.squeeze(outputs)

                outputs1 = torch.sign(outputs)
                #outputs1[:64][outputs1[:64] == -1] = 0
                acc = targets1.eq(outputs1)[:labeled_bs].sum().item()
                accuracy.append(acc)
            else:
                labeled_bs = data.size(0)

                labeled_loss = unlabeled_loss = torch.Tensor([0])
                outputs = torch.sign(outputs)
                outputs = torch.squeeze((outputs))
                outputs[outputs == -1] = 0
                acc = targets.eq(outputs).sum().item()
                loss = torch.mean(torch.pow(targets, outputs))
                loop_loss.append(loss.item() / len(data_loader))
                accuracy.append(acc)
                #loss = testAcc
                #print("test acc = {}".format(acc))



                #loss = torch.mean(self.loss_fn(outputs, targets))
            labeled_n += labeled_bs

            if print_freq>0 and (batch_idx%print_freq)==1:
                if mode == "train" and (self.epoch%10)==1:
                  fig, ax = plt.subplots(1, 1)
                  tsne = TSNE(n_components=2, random_state=0)
                  outputs1=outputs1.cpu()
                  label= np.where(outputs1 == -1)[0]
                  label1=np.where(outputs1 == 1)[0]
                  data1 = out[label, :]
                  data1=data1.cpu()
                  data2 = out[label1, :]
                  data2 = data2.cpu()
                  if data1.shape[0]>0 and data2.shape[0]>0:
                      data1 = data1.detach().numpy()
                      images_1d = tsne.fit_transform(data1)
                      ax.scatter(images_1d[:, 0], images_1d[:, 1], c='r', s=40, label='Class 1')
                      data2 = data2.detach().numpy()
                      images_2d = tsne.fit_transform(data2)
                      ax.scatter(images_2d[:, 0], images_2d[:, 1], c='b', s=40, label='Class 2')
                      ax.set_title(' ')
                      ax.legend()
                      plt.savefig("图片{}.png".format(self.epoch), dpi=150)
                      # plt.show()
                  targets = targets.to(self.device)
                print(f"[{mode}]loss[{batch_idx:<3}]\t labeled loss: {labeled_loss.item():.3f}\t unlabeled loss: {unlabeled_loss.item():.3f}\t loss: {loss.item():.3f}\t Acc: {acc/labeled_bs:.3%}")
            if self.writer:
                self.writer.add_scalar(mode+'_global_loss', loss.item(), self.global_step)
                self.writer.add_scalar(mode+'_global_accuracy', acc/labeled_bs, self.global_step)
        print(f">>>[{mode}]loss\t loss: {sum(loop_loss):.3f}\t Acc: {sum(accuracy)/labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar(mode+'_epoch_loss', sum(loop_loss), self.epoch)
            self.writer.add_scalar(mode+'_epoch_accuracy', sum(accuracy)/labeled_n, self.epoch)

        return loop_loss, accuracy

    def unlabeled_weight(self):
        alpha = 0.0
        if self.epoch > self.T1:
            alpha = (self.epoch-self.T1) / (self.T2-self.T1)*self.af
            if self.epoch > self.T2:
                alpha = af
        return alpha
        
    def train(self, data_loader, print_freq=20):
        self.model.train()
        with torch.enable_grad():
            loss, correct = self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            loss, correct = self._iteration(data_loader, print_freq, is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1):
        for ep in range(epochs):
            self.epoch = ep
            if scheduler is not None:
                scheduler.step()
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)
            print("------ Testing epochs: {} ------".format(ep))
            self.test(test_data, print_freq)
            if ep % self.save_freq == 0:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                    "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))
def creatLap(dist):
    Y, I = torch.sort(dist, dim=1)
    W = torch.zeros_like(dist)
    for ii in range(dist.size(0)):
        W[ii, I[ii, :60]] = 1
    B = W + W.t()
    W = dist.clone()
    mask = B < 0.9
    W = W * torch.logical_not(mask)
    t = 10
    W = torch.exp(-W.pow(2) / (2 * t * t))
    W_exp = W * torch.logical_not(mask)
    D = torch.sum(W_exp, dim=1)
    L = torch.diag(D) - W_exp
    return L