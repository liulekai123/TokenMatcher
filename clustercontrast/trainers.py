from __future__ import print_function, absolute_import
from audioop import cross
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import numpy as np
import os


##################
def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [B, m, d]
      y: pytorch Variable, with shape [B, n, d]
    Returns:
      dist: pytorch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x_norm = torch.pow(x, 2).sum(2, keepdim=True).sqrt().expand(B, m, n)
    y_norm = torch.pow(y, 2).sum(2, keepdim=True).sqrt().expand(B, n, m).transpose(-2, -1)
    xy_intersection = x @ y.transpose(-2, -1)
    dist = xy_intersection/(x_norm * y_norm)
    return torch.abs(dist)
class Dissimilar(object):
    def __init__(self, dynamic_balancer=True):
        self.dynamic_balancer = dynamic_balancer
    
    def __call__(self, features):
        B, N, C = features.shape
        dist_mat = cosine_dist(features, features)  # B*N*N
        # dist_mat = euclidean_dist(features, features)
        # 上三角index
        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]

        # 1.用softmax替换平均，使得相似度更高的权重更大
        if self.dynamic_balancer:
          weight = F.softmax(_dist, dim=-1)
          dist = torch.mean(torch.sum(weight*_dist, dim=1))
        # 2.直接平均
        else:
          dist = torch.mean(_dist, dim=(0, 1))
        return dist
########################

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_IR = memory
        self.memory_RGB = memory
        self.memory_ins_ir = memory
        self.memory_ins_rgb = memory
        
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400, i2r=None, r2i=None, 
              lamba_mate=0, lamba_cross=0, lamba_neighbor=0, topk_rgb=None, topk_ir=None, topk_r2i=None, topk_i2r=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            
            _, f_out_rgb, f_out_ir, SDC_rgb, SDC_ir = self._forward(inputs_rgb,inputs_ir,modal=0)

            # intra-modality nce loss
            loss_ir = self.memory_ir(f_out_ir, labels_ir)  
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
                
            dissimilar = Dissimilar(dynamic_balancer=True)
            Dissimilar_LOSS_ir = dissimilar(torch.stack(SDC_ir, dim=1))
            Dissimilar_LOSS_rgb = dissimilar(torch.stack(SDC_rgb, dim=1))
            Dissimilar_LOSS = Dissimilar_LOSS_ir + Dissimilar_LOSS_rgb
            
            # neighbor
            if lamba_neighbor!=0:
                topk_ir_ = [topk_ir[index.item()] for index in indexes_ir]
                topk_rgb_ = [topk_rgb[index.item()] for index in indexes_rgb]
                neighbor_ir = self.memory_ins_ir(f_out_ir, topk_ir_,indexes=indexes_ir) 
                neighbor_rgb = self.memory_ins_rgb(f_out_rgb, topk_rgb_,indexes=indexes_rgb)
                
                if epoch%2 == 0:
                    topk_r2i_ = [topk_r2i[index.item()] for index in indexes_rgb]
                    neighbor_r2i = self.memory_ins_ir(f_out_rgb, topk_r2i_,indexes=indexes_rgb)
                    loss_neighbor = (neighbor_ir + neighbor_rgb + neighbor_r2i) * lamba_neighbor
                else:
                    topk_i2r_ = [topk_i2r[index.item()] for index in indexes_ir]
                    neighbor_i2r = self.memory_ins_rgb(f_out_ir, topk_i2r_,indexes=indexes_ir) 
                    loss_neighbor = (neighbor_ir + neighbor_i2r + neighbor_rgb) * lamba_neighbor
                    
            else:
                loss_neighbor = torch.tensor(0)
            

            if r2i:
                if type(i2r[1]) == list:             
                    rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
                    ir2rgb_labels = torch.tensor([i2r[key.item()][0] for key in labels_ir]).cuda()   # !!!!!!!

                    cross_loss = 1 * self.memory_rgb(f_out_ir, ir2rgb_labels.long()) + 1 * self.memory_ir(f_out_rgb, rgb2ir_labels.long())
                    
                    ###############################    
                    if lamba_mate != 0:
                        label_rgb=[]
                        delete_idx = []
                        for index,label in enumerate(labels_rgb):
                            label_ir = rgb2ir_labels[index].item()    # 得到关联的ir标签
                            if len(i2r[label_ir]) == 2:
                                label_rgb0 = i2r[label_ir][0]         # 通过ir标签再找它关联的rgb0标签
                                label_rgb1 = i2r[label_ir][1]         # 通过ir标签再找它关联的rgb1标签
                                if epoch%2 == 0:                      # 交替loss
                                    if label.item() == label_rgb0:
                                        label_rgb.append(label_rgb1)
                                    else:
                                        delete_idx.append(index)
                                if epoch%2 == 1:
                                    if label.item() == label_rgb1:
                                        label_rgb.append(label_rgb0)
                                    else:
                                        delete_idx.append(index)
                            else:                                     # 如果ir标签只有一个对应的rgb标签,删除
                                delete_idx.append(index) 
                        labels_rgb = torch.tensor(label_rgb).cuda().long()
                        
                        remain_idx = list(set(range(f_out_rgb.shape[0])) - set(delete_idx))
                        if len(remain_idx) != 0:
                            f_out_rgb = f_out_rgb[remain_idx,:]
                            loss_mate = 1 * self.memory_RGB(f_out_rgb, labels_rgb)
                        else:
                            loss_mate = torch.tensor(0)
                    ##############################################       
                else:             
                    rgb2ir_labels = torch.tensor([r2i[key.item()][0] for key in labels_rgb]).cuda()
                    ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()  
                    
                    cross_loss = 1 * self.memory_rgb(f_out_ir, ir2rgb_labels.long()) + 1 * self.memory_ir(f_out_rgb, rgb2ir_labels.long())
                    
                    ###############################    
                    if lamba_mate != 0:
                        label_ir=[]
                        delete_idx = []
                        for index,label in enumerate(labels_ir):
                            label_rgb = ir2rgb_labels[index].item()    # 得到关联的ir标签
                            if len(r2i[label_rgb]) == 2:
                                label_ir0 = r2i[label_rgb][0]         # 通过ir标签再找它关联的rgb0标签
                                label_ir1 = r2i[label_rgb][1]         # 通过ir标签再找它关联的rgb1标签 
                                if epoch%2 == 0:
                                    if label.item() == label_ir0:
                                        label_ir.append(label_ir1)
                                    else:
                                        delete_idx.append(index)
                                if epoch%2 == 1:
                                    if label.item() == label_ir1:
                                        label_ir.append(label_ir0)
                                    else:
                                        delete_idx.append(index)
                            else:                                     # 如果ir标签只有一个对应的rgb标签,删除
                                delete_idx.append(index) 
                        labels_ir = torch.tensor(label_ir).cuda().long()
                        
                        remain_idx = list(set(range(f_out_ir.shape[0])) - set(delete_idx))
                        if len(remain_idx) != 0:
                            f_out_ir = f_out_ir[remain_idx,:]
                            loss_mate = 1 * self.memory_IR(f_out_ir, labels_ir)
                        else:
                            loss_mate = torch.tensor(0)
                    ##############################################    
            else:
                cross_loss = torch.tensor(0)
                loss_mate = torch.tensor(0)
            
            if lamba_mate == 0:
                loss_mate = torch.tensor(0)
            if lamba_neighbor == 0:
                loss_neighbor = torch.tensor(0)
            if lamba_cross == 0:
                cross_loss = torch.tensor(0)
            
            
            loss = loss_ir + loss_rgb + Dissimilar_LOSS + loss_neighbor + lamba_cross*cross_loss + lamba_mate*loss_mate   #####################
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item()) 

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'loss Dissimilar {:.3f}\t'
                      'loss neighbor {:.3f}\t'
                      'Loss cross {:.3f}\t'
                      'Loss mate {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,losses.val, losses.avg,
                              loss_ir,loss_rgb,Dissimilar_LOSS,loss_neighbor,cross_loss,loss_mate,
                              ))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, indexes, _ = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, indexes, _ = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()
    
    def _forward(self, x1, x2, modal=0):
        return self.encoder(x1, x2, modal)
        


class ClusterContrastTrainer_cam(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_cam, self).__init__()
        self.encoder = encoder
        self.memory = memory
        
    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, mode=1):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()


        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            inputs, labels, indexes = self._parse_data(inputs)

            _, f_out_rgb, f_out_ir, SDC_rgb, SDC_ir = self._forward(inputs,inputs,modal=mode)
            

            # intra-modality nce loss
            loss_cam = self.memory(f_out_rgb, labels) 

            dissimilar = Dissimilar(dynamic_balancer=True)
            Dissimilar_LOSS = dissimilar(torch.stack(SDC_rgb, dim=1))
            
            loss = loss_cam + Dissimilar_LOSS   #####################
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item()) 

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss cam {:.3f}\t'
                      'Dissimilar LOSS {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_cam, Dissimilar_LOSS
                              ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()
    
    def _forward(self, x1, x2, modal=0):
        return self.encoder(x1, x2, modal)

    #def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
    #    return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

