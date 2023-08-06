'''
Date: 2022-07-20 15:36:44
LastEditors: yuhhong
LastEditTime: 2022-11-11 23:29:18
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random



def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# class ChamferLoss(nn.Module):

#     def __init__(self):
#         super(ChamferLoss, self).__init__()
#         self.use_cuda = torch.cuda.is_available()

#     def forward(self, preds, gts):
#         P = self.batch_pairwise_dist(gts, preds)
#         mins, _ = torch.min(P, 1)
#         loss_1 = torch.sum(mins)
#         mins, _ = torch.min(P, 2)
#         loss_2 = torch.sum(mins)
#         return loss_1 + loss_2

#     def batch_pairwise_dist(self, x, y):
#         bs, num_points_x, points_dim = x.size()
#         _, num_points_y, _ = y.size()
#         xx = torch.bmm(x, x.transpose(2, 1))
#         yy = torch.bmm(y, y.transpose(2, 1))
#         zz = torch.bmm(x, y.transpose(2, 1))
#         if self.use_cuda:
#             dtype = torch.cuda.LongTensor
#         else:
#             dtype = torch.LongTensor
#         diag_ind_x = torch.arange(0, num_points_x).type(dtype)
#         diag_ind_y = torch.arange(0, num_points_y).type(dtype)
#         rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
#             zz.transpose(2, 1))
#         ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
#         P = rx.transpose(2, 1) + ry - 2 * zz
#         return P

# refer: 
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
# class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

#     def __init__(self, optimizer, warmup, max_iters):
#         self.warmup = warmup
#         self.max_num_iters = max_iters
#         super().__init__(optimizer)

#     def get_lr(self):
#         lr_factor = self.get_lr_factor(epoch=self.last_epoch)
#         return [base_lr * lr_factor for base_lr in self.base_lrs]

#     def get_lr_factor(self, epoch):
#         lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
#         if epoch <= self.warmup:
#             lr_factor *= epoch * 1.0 / self.warmup
#         return lr_factor

# refer: 
# https://zhuanlan.zhihu.com/p/75542467
# Focal Loss for binary classification
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
#         super(FocalLoss, self).__init__()
#         self.alpha = torch.tensor(alpha).cuda()
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, pred, target):
#         # pred = nn.Sigmoid()(pred)
#         pred = pred.view(-1, 1)
#         target = target.view(-1, 1)
#         pred = torch.cat((1-pred,pred), dim=1)
        
#         class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
#         # scatter_(dim,index,src)->Tensor
#         # Writes all values from the tensor src into self at the indices specified in the index tensor. 
#         # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
#         class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

#         probs = (pred * class_mask).sum(dim=1).view(-1, 1)
#         probs = probs.clamp(min=0.0001, max=1.0)

#         log_p = probs.log()

#         alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
#         alpha[:,0] = alpha[:,0] * (1-self.alpha)
#         alpha[:,1] = alpha[:,1] * self.alpha
#         alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

#         if self.reduction == "mean":
#             loss = batch_loss.mean()
#         elif self.reduction == "sum":
#             loss = batch_loss.sum()
        
#         return loss


"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    assert labels.get_device() == logits.get_device()
    device = labels.get_device()
    device = torch.device("cuda:" + str(device)) if device >= 0 else torch.device("cpu")

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    
    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weights=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weights=weights)
    return cb_loss