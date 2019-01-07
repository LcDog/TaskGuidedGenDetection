# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import copy
import time
import cv2
from PIL import Image

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

sys.path.append(os.path.join(os.getcwd(), 'lib'))
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.config_eval import cfg_eval, cfg_from_file_eval, cfg_from_list_eval, get_output_dir_eval
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from tensorboardX import SummaryWriter

from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='sim10k', type=str)
  parser.add_argument('--eval_dataset', 
                      default='cityscapes', type=str)

  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                      default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=2000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="./data/out",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=8, type=int)
  parser.add_argument('--cuda', dest='cuda', type=int, default=1)
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=4, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-2, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=8, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  parser.add_argument('--data_path', type=str,
                      default='/mnt/data-3/data/liangchen.song/det')
  parser.add_argument('--model_path_prefix', type=str,
                      default='/mnt/data-3/data/liangchen.song/models')
  parser.add_argument('--eval_interval', type=int, default=1000)
  parser.add_argument('--vis', default=1, type=int)
  parser.add_argument('--tensorboard_log_dir', default='./logs/', type=str)
  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


def init_eval_cfg(args):
  tem_args = copy.deepcopy(args)

  np.random.seed(cfg_eval.RNG_SEED)
  if tem_args.eval_dataset == "cityscapes":
      tem_args.imdbval_name = "cityscapes_val"
      tem_args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  tem_args.cfg_file = "cfgs/{}_ls.yml".format(tem_args.net) if tem_args.large_scale else "cfgs/{}.yml".format(tem_args.net)

  if tem_args.cfg_file is not None:
    cfg_from_file_eval(tem_args.cfg_file)
  if tem_args.set_cfgs is not None:
    cfg_from_list_eval(tem_args.set_cfgs)

  # print('Using config:')
  # pprint.pprint(cfg_eval)

  if tem_args.cuda:
    cfg_eval.CUDA = True

  cfg_eval.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(tem_args.imdbval_name, False)
  imdb.competition_mode(on=True)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)
  return imdb, dataloader


# def evaluate(model, args, writer=None, step=0):
def evaluate(model, args, imdb, dataloader, writer=None, step=0):
  # cfg_eval = copy.deepcopy(cfg)
  tem_args = copy.deepcopy(args)

  # np.random.seed(cfg_eval.RNG_SEED)
  # if tem_args.eval_dataset == "pascal_voc":
  #     tem_args.imdb_name = "voc_2007_trainval"
  #     tem_args.imdbval_name = "voc_2007_test"
  #     tem_args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif tem_args.eval_dataset == "pascal_voc_0712":
  #     tem_args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
  #     tem_args.imdbval_name = "voc_2007_test"
  #     tem_args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif tem_args.eval_dataset == "coco":
  #     tem_args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
  #     tem_args.imdbval_name = "coco_2014_minival"
  #     tem_args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif tem_args.eval_dataset == "imagenet":
  #     tem_args.imdb_name = "imagenet_train"
  #     tem_args.imdbval_name = "imagenet_val"
  #     tem_args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif tem_args.eval_dataset == "vg":
  #     tem_args.imdb_name = "vg_150-50-50_minitrain"
  #     tem_args.imdbval_name = "vg_150-50-50_minival"
  #     tem_args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  # elif tem_args.eval_dataset == "cityscapes":
  #     tem_args.imdbval_name = "cityscapes_val"
  #     tem_args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  # tem_args.cfg_file = "cfgs/{}_ls.yml".format(tem_args.net) if tem_args.large_scale else "cfgs/{}.yml".format(tem_args.net)

  # if tem_args.cfg_file is not None:
  #   cfg_from_file_eval(tem_args.cfg_file)
  # if tem_args.set_cfgs is not None:
  #   cfg_from_list_eval(tem_args.set_cfgs)

  # # print('Using config:')
  # # pprint.pprint(cfg_eval)

  # cfg_eval.TRAIN.USE_FLIPPED = False
  # imdb, roidb, ratio_list, ratio_index = combined_roidb(tem_args.imdbval_name, False)
  # imdb.competition_mode(on=True)

  # print('{:d} roidb entries'.format(len(roidb)))
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if tem_args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  start = time.time()
  max_per_image = 100

  vis = tem_args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
              for _ in xrange(imdb.num_classes)]

  # output_dir = get_output_dir(imdb, save_name)
  output_dir = './data/out'
  # dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
  #                       imdb.num_classes, training=False, normalize = False)
  # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
  #                           shuffle=False, num_workers=0,
  #                           pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  # det_file = os.path.join(output_dir, load_name_file+'_detections.pkl')
  # fasterRCNN = model.module

  model.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = model(im_data, im_info, gt_boxes, num_boxes)

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      if cfg_eval.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg_eval.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if tem_args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg_eval.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                          + torch.FloatTensor(cfg_eval.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg_eval.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                          + torch.FloatTensor(cfg_eval.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      for j in xrange(1, imdb.num_classes):
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if tem_args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg_eval.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
      #     .format(i + 1, num_images, detect_time, nms_time))
      # sys.stdout.flush()
      if (i + 1) % 100 == 0:
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
        if writer is not None:
          im2show = im2show.transpose((2,0,1))
          im2show = im2show[:,:,::-1]
          writer.add_image('det_result', im2show, step)

      # if vis:
          # cv2.imwrite('result.png', im2show)
          # pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  # with open(det_file, 'wb') as f:
  #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  miou05 = imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
  return miou05


if __name__ == '__main__':

  args = parse_args()
  cfg.DATA_DIR = args.data_path
  writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
  highest = 0

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "sim10k":
      args.imdb_name = "sim10k"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  # output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  # if not os.path.exists(output_dir):
  #   os.makedirs(output_dir)
  output_dir = args.save_dir

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  imdb_eval, dataloader_eval = init_eval_cfg(args)  

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, model_path_prefix=args.model_path_prefix, cfg=cfg)
    fasterRCNN_eval = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic, model_path_prefix=args.model_path_prefix, cfg=cfg_eval)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  fasterRCNN_eval.create_architecture()
  for params in fasterRCNN_eval.parameters():
    params.requires_grad = False

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN.cuda()
    fasterRCNN_eval.cuda()

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      # setting to train mode
      fasterRCNN.train()

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()


      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        # if args.use_tfboard:
        info = {
          'loss': loss_temp,
          'loss_rpn_cls': loss_rpn_cls,
          'loss_rpn_box': loss_rpn_box,
          'loss_rcnn_cls': loss_rcnn_cls,
          'loss_rcnn_box': loss_rcnn_box
        }
        writer.add_scalars("losses", info, (epoch - 1) * iters_per_epoch + step)
          
        loss_temp = 0
        start = time.time()

      if (step+1) % args.eval_interval == 0:
        fasterRCNN_eval.load_state_dict(fasterRCNN.module.state_dict())
        print('-'*70)
        print('-'*70)
        print('-'*70)
        miou05 = evaluate(fasterRCNN_eval, args, imdb_eval, dataloader_eval, writer, step)
        if miou05 > highest:
          torch.save(fasterRCNN.module.state_dict(), os.path.join(output_dir, 'highest.pth'))
          print('>'*20 + f'save highest with {miou05:.2%}')
          highest = miou05
        print('-'*70)
        print('-'*70)
        print('-'*70)

    # save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    # save_checkpoint({
    #   'session': args.session,
    #   'epoch': epoch + 1,
    #   'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
    #   'optimizer': optimizer.state_dict(),
    #   'pooling_mode': cfg.POOLING_MODE,
    #   'class_agnostic': args.class_agnostic,
    # }, save_name)
    # print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
  
  writer.close()
