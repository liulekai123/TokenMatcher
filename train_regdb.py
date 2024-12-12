# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer_camTokens,ClusterContrastTrainer_Tokens,ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter
start_epoch = best_mAP = 0

def get_data(name, data_dir,trial=0):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root,trial=trial)
    return dataset


class channel_select(object):
    def __init__(self,channel=0):
        self.channel = channel

    def __call__(self, img):
        if self.channel == 3:
            img_gray = img.convert('L')
            np_img = np.array(img_gray, dtype=np.uint8)
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        else:
            np_img = np.array(img, dtype=np.uint8)
            np_img = np_img[:,:,self.channel]
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        return img_PIL



def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):


    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):



    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    if args.arch == 'vit_base':
        print('vit_base')
        model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate
                , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem)
    else:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model




class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_gall_feat(model,gall_loader,ngall):
    pool_dim=768*cls_token_num   ######2048 768 3072
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input,input, 2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc
    
def extract_query_feat(model,query_loader,nquery):
    pool_dim=768*cls_token_num
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input, input,1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc



def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()




def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)
def eval_regdb(distmat, q_pids, g_pids, max_rank = 20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    log_s1_name = 'regdb_s1'
    log_s2_name = 'regdb_s2'
    main_worker_stage1(args,log_s1_name)#Stage 1
    main_worker_stage2(args,log_s1_name,log_s2_name) #Stage 2


def main_worker_stage1(args,log_s1_name):
    logs_dir_root = osp.join(args.logs_dir+'/'+log_s1_name)
    trial = args.trial
    global cls_token_num
    cls_token_num=args.cls_token_num
    start_epoch =0
    best_mAP =0
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir,trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir,trial=trial)

    # Create model
    model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate0
            , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem,cls_token_num=cls_token_num)
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    #optimizer = torch.optim.Adam(params, lr=args.lr0, weight_decay=args.weight_decay0)  #!!!!!!!!!!!!!!!!!!
    optimizer = torch.optim.SGD(params, lr=args.lr0, momentum=0.9, weight_decay=args.weight_decay0)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-7)

    # Trainer
    trainer = ClusterContrastTrainer(model)
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.2
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.2
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_=F.normalize(features_rgb, dim=1)
            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_=F.normalize(features_ir, dim=1)
            
            rerank_dist_ir = compute_jaccard_distance(features_ir_, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            del rerank_dist_ir

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)        
        
        rerank_dist_rgb = compute_jaccard_distance(features_rgb_, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
        pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
        del rerank_dist_rgb
        
        num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
        num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        memory_ir = ClusterMemory(model.module.num_features*cls_token_num, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum0, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(model.module.num_features*cls_token_num, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum0, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        pseudo_labeled_dataset_ir = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ########################
        color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)#T.
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
          color_aug,
          T.Resize((height, width)),#, interpolation=3
          T.Pad(10),
          T.RandomCrop((height, width)),
          T.RandomHorizontalFlip(p=0.5),
          T.ToTensor(),
          normalizer,
          ChannelRandomErasing(probability = 0.5)
          ])
    
        train_transformer_rgb1 = T.Compose([
          color_aug,
          T.Resize((height, width)),
          T.Pad(10),
          T.RandomCrop((height, width)),
          T.RandomHorizontalFlip(p=0.5),
          T.ToTensor(),
          normalizer,
          ChannelRandomErasing(probability = 0.5),
          ChannelExchange(gray = 2)
          ])
    
        transform_thermal = T.Compose( [
            color_aug,
            T.Resize((height, width)),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)
            ])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        (args.batch_size//2), args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

        if epoch>=6 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path='/scratch/chenjun3/liulekai/ADCA/data/regdb/'
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model,gall_loader,ngall)
            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)


            print('Test Trial: {}'.format(trial))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' ****************' if is_best else ''))
############################
        lr_scheduler.step()
        print("the learning rate is ", optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
    
    fpath = osp.join(args.logs_dir, 'checkpoint.pth.tar')
    if os.path.isfile(fpath):
        os.remove(fpath)
        print(f"文件 {fpath} 已删除。")
    else:
        print(f"文件 {fpath} 不存在。")


def main_worker_stage2(args,log_s1_name,log_s2_name):
    print(log_s2_name)
    global cls_token_num
    cls_token_num=args.cls_token_num
    logs_dir_root = osp.join('logs/'+log_s2_name)
    trial = args.trial
    # global start_epoch, best_mAP
    start_epoch =0
    best_mAP =0
    args.logs_dir = osp.join(logs_dir_root,str(trial))
    start_time = time.monotonic()

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial)+'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir,trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir,trial=trial)

    # Create model
    #model = create_model(args)
    model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate1
            , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem,cls_token_num=cls_token_num)
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    
    checkpoint = load_checkpoint(osp.join('./logs/'+log_s1_name+'/'+str(trial), 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    #optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(params, lr=args.lr1, momentum=0.9, weight_decay=args.weight_decay1) 
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-7)
    
    # Trainer 
    trainer = ClusterContrastTrainer(model)   #########################
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.2
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.2
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_=F.normalize(features_rgb, dim=1)
            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_=F.normalize(features_ir, dim=1)

            rerank_dist_ir = compute_jaccard_distance(features_ir_, k1=args.k1, k2=args.k2,search_option=3)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            del rerank_dist_ir
            

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        rerank_dist_rgb = compute_jaccard_distance(features_rgb_, k1=args.k1, k2=args.k2,search_option=3)
        pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
        del rerank_dist_rgb
        num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
        num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        
        memory_ir = ClusterMemory(768*cls_token_num, num_cluster_ir, temp=args.temp, momentum=0.2, use_hard=args.use_hard)   # 0.2
        memory_rgb = ClusterMemory(768*cls_token_num, num_cluster_rgb, temp=args.temp, momentum=0.2, use_hard=args.use_hard)
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        trainer.memory_IR = memory_ir
        trainer.memory_RGB = memory_rgb

        pseudo_labeled_dataset_ir = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))
        
        ############################################# 邻居学习：各个token的邻居和cam的邻居的交集       
        num_ir_imgs = dataset_ir.num_train_imgs
        num_rgb_imgs = dataset_rgb.num_train_imgs
        memory_instance_ir = ClusterMemory(768*cls_token_num, num_ir_imgs, temp=args.temp, momentum=args.momentum0, use_hard=args.use_hard)
        memory_instance_rgb = ClusterMemory(768*cls_token_num, num_rgb_imgs, temp=args.temp, momentum=args.momentum0, use_hard=args.use_hard)
        memory_instance_ir.features = F.normalize(features_ir, dim=1).cuda()
        memory_instance_rgb.features = F.normalize(features_rgb, dim=1).cuda()
        trainer.memory_ins_ir = memory_instance_ir
        trainer.memory_ins_rgb = memory_instance_rgb
                
        def topk(matrix,k):
            matrix = np.array(matrix)
            row_max = np.max(matrix, axis=1)
            threshold = k * row_max[:, np.newaxis]
            indices = [list(np.where(row > th)[0]) for row, th in zip(matrix, threshold)]  # numpy
            return indices
        def hebing(list1,list2):
            merged_list = []
            for i,(row1, row2) in enumerate(zip(list1, list2)):
                merged_list.append(row1 + row2)   # numpy 的+表示对应元素相加
            return merged_list
        def sort_by_frequency_unique(row, X):
            count = Counter(row)
            sorted_elements = [element for element, freq in count.items() if freq >= X]
            return sorted_elements
                
        features_cls_ir = features_ir_[:, 0:768]
        features_cls_rgb = features_rgb_[:, 0:768]
        topk_ir = topk(torch.mm(features_cls_ir, features_cls_ir.T), args.k)
        topk_rgb = topk(torch.mm(features_cls_rgb, features_cls_rgb.T), args.k)
        topk_i2r = topk(torch.mm(features_cls_ir, features_cls_rgb.T), args.k)
        topk_r2i = topk(torch.mm(features_cls_rgb, features_cls_ir.T), args.k)
        for i in range(cls_token_num-1):
            features_cls_ir = features_ir_[:, (i+1)*768:(i+2)*768]
            features_cls_rgb = features_rgb_[:, (i+1)*768:(i+2)*768]
            topk_ir_ = topk(torch.mm(features_cls_ir, features_cls_ir.T), args.k)
            topk_rgb_ = topk(torch.mm(features_cls_rgb, features_cls_rgb.T), args.k)
            topk_i2r_ = topk(torch.mm(features_cls_ir, features_cls_rgb.T), args.k)
            topk_r2i_ = topk(torch.mm(features_cls_rgb, features_cls_ir.T), args.k)
            topk_rgb = hebing(topk_rgb,topk_rgb_)
            topk_ir = hebing(topk_ir,topk_ir_)
            topk_i2r = hebing(topk_i2r,topk_i2r_)
            topk_r2i = hebing(topk_r2i,topk_r2i_)
        topk_rgb = [sort_by_frequency_unique(row, args.x) for row in topk_rgb]
        topk_ir = [sort_by_frequency_unique(row, args.x) for row in topk_ir]
        topk_i2r = [sort_by_frequency_unique(row, args.x) for row in topk_i2r]
        topk_r2i = [sort_by_frequency_unique(row, args.x) for row in topk_r2i]
        del features_cls_rgb,features_cls_ir,topk_ir_,topk_rgb_,topk_i2r_,topk_r2i_,features_ir_,features_rgb_
        print(len(topk_rgb[0]),len(topk_ir[0]),len(topk_i2r[0]),len(topk_r2i[0]))
        ##############################
        
        ######################################  R2I
        cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
        cluster_features_ir = F.normalize(cluster_features_ir, dim=1)
        # 全局的最相似最优先匹配，若该最小值的ir标签已经被分配，如果这个ir标签只被分配有一个rgb，则该最小值分配给这个ir标签。
        #                                                      如果这个ir标签被分配2个rgb，则该最小值跳过。
        print("Matching!!!")
        if num_cluster_rgb >= num_cluster_ir:
            # clusternorm
            similarity = ((torch.mm(cluster_features_rgb, cluster_features_ir.T))/1).exp().cpu() #.exp().cpu()
            cost = similarity / 1   # 越大越相似，tensor
            cost_1 = cost.clone()
            
            r2i = {}       # rgb实例序号->ir簇伪标签
            i2r = {}
            col = torch.zeros(num_cluster_ir)       # 列是否被分配，0表示未分配，1表示已分配
            for j in range(num_cluster_rgb*num_cluster_ir):
                if len(r2i) == num_cluster_rgb:
                    break
                max_index = np.argmax(cost)         # 返回顺着数的序号，返回tensor!!
                row_index, col_index = np.unravel_index(max_index, cost.shape)  # 返回int
                if col[col_index] == 0 :            # 如果这个最小值的列没有被分配, 则选定
                    r2i[row_index] = col_index
                    i2r[col_index] = [row_index]
                    col[col_index] = 1        
                    cost[row_index, :] = -1000       
                elif col[col_index] < args.lamba_k:           # 这个ir标签只被分配有一个rgb，则该最小值分配给这个ir标签。
                    r2i[row_index] = col_index
                    i2r[col_index].append(row_index)
                    col[col_index] += 1        
                    cost[row_index, :] = -1000
                else:                               # 这个ir标签分配了两个rgb，跳过
                    cost[row_index, col_index] = -1000
            
            # 可能有的ir标签没有被分配
            for idx_ir,item in enumerate(col):
                if item != 0:
                    cost_1[:,idx_ir] = -1000
            cost = cost_1
            del cost_1

            # 最后还是一个ir配多个rgb
            for j in range(num_cluster_rgb*num_cluster_ir):
                if len(i2r) == num_cluster_ir:
                    break
                max_index = np.argmax(cost)         # 返回顺着数的序号，返回tensor!!
                row_index, col_index = np.unravel_index(max_index, cost.shape)  # 返回int
                if col[col_index] == 0 :            # 如果这个最小值的列没有被分配, 则选定
                    i2r[col_index] = [row_index]
                    col[col_index] = 1        
                    cost[row_index, :] = -1000       
                elif col[col_index] < args.lamba_k:           # 这个ir标签只被分配有一个rgb，则该最小值分配给这个ir标签。
                    i2r[col_index].append(row_index)
                    col[col_index] += 1        
                    cost[row_index, :] = -1000
                else:                               # 这个ir标签分配了两个rgb，跳过
                    cost[row_index, col_index] = -1000
        
        if num_cluster_ir > num_cluster_rgb:
            # clusternorm
            similarity = ((torch.mm(cluster_features_ir, cluster_features_rgb.T))/1).exp().cpu() #.exp().cpu()
            cost = similarity / 1   # 越小越相似，tensor
            cost_1 = cost.clone()
            
            r2i = {}       # rgb实例序号->ir簇伪标签
            i2r = {}
            col = torch.zeros(num_cluster_rgb)       # 列是否被分配，0表示未分配，1表示已分配
            for j in range(num_cluster_rgb*num_cluster_ir):
                if len(i2r) == num_cluster_ir:
                    break
                max_index = np.argmax(cost)         # 返回顺着数的序号，返回tensor!!
                row_index, col_index = np.unravel_index(max_index, cost.shape)  # 返回int
                if col[col_index] == 0 :            # 如果这个最小值的列没有被分配, 则选定
                    i2r[row_index] = col_index
                    r2i[col_index] = [row_index]
                    col[col_index] = 1        
                    cost[row_index, :] = -1000       
                elif col[col_index] < args.lamba_k:           # 这个ir标签只被分配有一个rgb，则该最小值分配给这个ir标签。
                    i2r[row_index] = col_index
                    r2i[col_index].append(row_index)
                    col[col_index] += 1        
                    cost[row_index, :] = -1000
                else:                               # 这个ir标签分配了两个rgb，跳过
                    cost[row_index, col_index] = -1000

            # 可能有的rgb标签没有被分配
            for idx_ir,item in enumerate(col):
                if item != 0:
                    cost_1[:,idx_ir] = -1000
            cost = cost_1
            del cost_1

            for j in range(num_cluster_rgb*num_cluster_ir):
                if len(r2i) == num_cluster_rgb:
                    break
                max_index = np.argmax(cost)         # 返回顺着数的序号，返回tensor!!
                row_index, col_index = np.unravel_index(max_index, cost.shape)  # 返回int
                if col[col_index] == 0 :            # 如果这个最小值的列没有被分配, 则选定
                    r2i[col_index] = [row_index]
                    col[col_index] = 1        
                    cost[row_index, :] = -1000       
                elif col[col_index] < args.lamba_k:           # 这个ir标签只被分配有一个rgb，则该最小值分配给这个ir标签。
                    r2i[col_index].append(row_index)
                    col[col_index] += 1        
                    cost[row_index, :] = -1000
                else:                               # 这个ir标签分配了两个rgb，跳过
                    cost[row_index, col_index] = -1000
        
        del cost, similarity, col
        print("Matching Done!!!")
        ####################################################
        
        features_all = torch.cat((features_rgb,features_ir),dim=0)
        pseudo_labels_rgb = torch.from_numpy(pseudo_labels_rgb)
        pseudo_labels_rgb_ = pseudo_labels_rgb.clone()
        for i,label in enumerate(pseudo_labels_rgb):
            if label != -1:
                if num_cluster_rgb >= num_cluster_ir:
                    pseudo_labels_rgb[i] = r2i[label.item()]
                else:
                    pseudo_labels_rgb[i] = r2i[label.item()][0]
        pseudo_labels_all = torch.cat((pseudo_labels_rgb,torch.from_numpy(pseudo_labels_ir)),dim=-1).view(-1).cpu().numpy()
        cluster_features_ir = generate_cluster_features(pseudo_labels_all, features_all)
        memory_mixIR = ClusterMemory(768*cls_token_num, num_cluster_ir, temp=args.temp,momentum=args.momentum1, use_hard=args.use_hard)#.cuda()
        memory_mixIR.features = F.normalize(cluster_features_ir, dim=1).cuda()
        trainer.memory_ir = memory_mixIR 
        
        for i,label in enumerate(pseudo_labels_ir):
            if label != -1:
                if num_cluster_rgb >= num_cluster_ir:
                    pseudo_labels_ir[i] = i2r[label.item()][0]
                else:
                    pseudo_labels_ir[i] = i2r[label.item()]
        pseudo_labels_all = torch.cat((pseudo_labels_rgb_,torch.from_numpy(pseudo_labels_ir)),dim=-1).view(-1).cpu().numpy()
        cluster_features_rgb = generate_cluster_features(pseudo_labels_all, features_all)
        memory_mixRGB = ClusterMemory(768*cls_token_num, num_cluster_rgb, temp=args.temp,momentum=args.momentum1, use_hard=args.use_hard)#.cuda()
        memory_mixRGB.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        trainer.memory_rgb = memory_mixRGB
        del cluster_features_ir, cluster_features_rgb
        #################################################################
        
        color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)#T.
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
          color_aug,
          T.Resize((height, width)),#, interpolation=3
          T.Pad(10),
          T.RandomCrop((height, width)),
          T.RandomHorizontalFlip(p=0.5),
          T.ToTensor(),
          normalizer,
          ChannelRandomErasing(probability = 0.5)
          ])
    
        train_transformer_rgb1 = T.Compose([
          color_aug,
          T.Resize((height, width)),
          T.Pad(10),
          T.RandomCrop((height, width)),
          T.RandomHorizontalFlip(p=0.5),
          T.ToTensor(),
          normalizer,
          ChannelRandomErasing(probability = 0.5),
          ChannelExchange(gray = 2)
          ])
    
        transform_thermal = T.Compose( [
            color_aug,
            T.Resize((height, width)),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)
            ])
        #############################
        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        (args.batch_size//2), args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i, 
                    lamba_cross=args.lamba_cross, lamba_mate=args.lamba_mate, lamba_neighbor=args.lamba_neighbor, 
                    topk_rgb=topk_rgb,topk_ir=topk_ir,topk_r2i=topk_r2i,topk_i2r=topk_i2r) ###############
        del train_loader_ir, train_loader_rgb,topk_rgb,topk_ir,topk_r2i,topk_i2r

        if epoch>=6 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])

            data_path='/scratch/chenjun3/liulekai/ADCA/data/regdb/'
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model,gall_loader,ngall)
            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)


            print('Test Trial: {}'.format(trial))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            is_best = (cmc[0] > best_mAP)
            best_mAP = max(cmc[0], best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model r1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, cmc[0], best_mAP, ' **********************' if is_best else ''))
############################
        lr_scheduler.step()
        print("the learning rate is ", optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')
        
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
    
    fpath = osp.join(args.logs_dir, 'checkpoint.pth.tar')
    # 检查文件是否存在，然后删除它
    if os.path.isfile(fpath):
        os.remove(fpath)
        print(f"文件 {fpath} 已删除。")
    else:
        print(f"文件 {fpath} 不存在。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('-pp', '--pretrained-path', type=str, default='')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum0', type=float, default=0.8,
                        help="update momentum for the hybrid memory0")
    parser.add_argument('--momentum1', type=float, default=0.8,
                        help="update momentum for the hybrid memory1")
    #vit
    parser.add_argument('--drop-path-rate0', type=float, default=0.3)
    parser.add_argument('--drop-path-rate1', type=float, default=0.3)
    parser.add_argument('--hw-ratio', type=int, default=2)
    parser.add_argument('--self-norm', action="store_true")
    parser.add_argument('--conv-stem', action="store_true")
    # optimizer
    parser.add_argument('--lr0', type=float, default=0.000035,
                        help="learning rate0")
    parser.add_argument('--lr1', type=float, default=0.000035,
                        help="learning rate1")
    parser.add_argument('--weight-decay0', type=float, default=5e-4)
    parser.add_argument('--weight-decay1', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--step-size', type=int, default=15)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default="/scratch/chenjun3/liulekai/ADCA/data")
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    
    parser.add_argument('--lamba-mate', type=float, default=0.01)  #########################
    parser.add_argument('--lamba-cross', type=float, default=0.4)
    parser.add_argument('--lamba-neighbor', type=float, default=0.1)
    parser.add_argument('--cls-token-num', type=int, default=6)
    parser.add_argument('--k', type=float, default=0.9)
    parser.add_argument('--x', type=int, default=6)
    parser.add_argument('--lamba-k', type=int, default=2)

    main()
