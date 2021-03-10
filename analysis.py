import numpy as np
import json
import glob
#import cv2
import os
import pdb
import scipy.misc as misc
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import logging
import time

logger = logging.getLogger("Model")

def scores_unct(label_trues, label_preds, unct, n_class, percent):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    mask = unct < np.sort(unct)[int((unct.shape[0] - 1) * percent)]
    # print 'percent', percent, 'is', np.sort(unct)[int((unct.shape[0]-1)*percent)]
    label_trues = label_trues[mask]
    label_preds = label_preds[mask]
    hist = np.zeros((n_class, n_class))
    hist += _fast_hist(label_trues, label_preds, n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu, }, cls_iu


def PR_Curve(args, pred_names, unct_names, label_names, acqu_func):
    x = np.array(range(1, 101))
    global_acc = []
    mean_acc = []
    freqw_acc = []
    Mean_IOU = []

    labels = np.array([], dtype='uint8')
    preds = np.array([], dtype='uint8')
    uncts = np.array([], dtype='uint8')
    for index, pred_name in enumerate(pred_names):
        pred = np.load(pred_name)
        unct = np.load(unct_names[index])
        label = cv2.imread(label_names[index])[..., 0]
        labels = np.append(labels, label.flatten())
        preds = np.append(preds, pred.flatten())
        uncts = np.append(uncts, unct.flatten())
    for i in range(1, 101):
        score, class_iou = scores_unct(labels, preds, uncts, 11, i / 100.0)
        # print i, '%' 'Overall Acc: \t', score['Overall Acc: \t'], 'Mean IoU : \t', score['Mean IoU : \t']
        global_acc.append(score['Overall Acc: \t'])
        mean_acc.append(score['Mean Acc : \t'])
        freqw_acc.append(score['FreqW Acc : \t'])
        Mean_IOU.append(score['Mean IoU : \t'])

    print('global_acc:'+str(sum(global_acc)))
    print('mean_acc:'+str(sum(mean_acc)))
    print('freqw_acc:'+str(sum(freqw_acc)))
    print('Mean_IOU:'+str(sum(Mean_IOU)))

    f = open(os.path.join(args.out_dir, 'pixel_level_PR_curve_' + acqu_func + '.txt'), 'w')
    f.write(
        'global_acc\n' + str(global_acc) + ' ' + str(sum(global_acc)) + '\n\nmean_acc\n' + str(mean_acc) + ' ' + str(
            sum(mean_acc)) + \
        '\n\nfreqw_acc\n' + str(freqw_acc) + ' ' + str(sum(freqw_acc)) + '\n\nMean_IOU\n' + str(Mean_IOU) + ' ' + str(
            sum(Mean_IOU)))
    f.close()

    return global_acc, mean_acc, freqw_acc, Mean_IOU

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


### 参考efficient-uncert··· validate.py
def output_pred_unct(args,experiment_dir,out_name, label, pred, unct_r, unct_e,unct_b, unct_v):
    out_label_dir = os.path.join(experiment_dir, 'output_val', 'label')
    out_pred_dir = os.path.join(experiment_dir, 'output_val', 'pred')
    #out_preds_dir = os.path.join(experiment_dir, 'output_val', 'preds')
    out_unct_r_dir = os.path.join(experiment_dir, 'output_val', 'unct_r')
    out_unct_e_dir = os.path.join(experiment_dir, 'output_val', 'unct_e')
    out_unct_b_dir = os.path.join(experiment_dir, 'output_val', 'unct_b')
    out_unct_v_dir = os.path.join(experiment_dir, 'output_val', 'unct_v')
    if not os.path.exists(out_label_dir):
        os.makedirs(out_label_dir)
    if not os.path.exists(out_pred_dir):
        os.makedirs(out_pred_dir)
    '''
    if not os.path.exists(out_preds_dir):
        os.makedirs(out_preds_dir)
    '''
    if not os.path.exists(out_unct_r_dir):
        os.makedirs(out_unct_r_dir)
    if not os.path.exists(out_unct_e_dir):
        os.makedirs(out_unct_e_dir)
    if not os.path.exists(out_unct_b_dir):
        os.makedirs(out_unct_b_dir)
    if not os.path.exists(out_unct_v_dir):
        os.makedirs(out_unct_v_dir)

    if args.save_output:
        print("----"+out_name+': outputing'+"----")
        np.save(os.path.join(out_label_dir, out_name), label)
        np.save(os.path.join(out_pred_dir, out_name), pred)
        #np.save(os.path.join(out_preds_dir, out_name), preds)
        np.save(os.path.join(out_unct_r_dir, out_name), unct_r)
        np.save(os.path.join(out_unct_e_dir, out_name), unct_e)
        np.save(os.path.join(out_unct_b_dir, out_name), unct_b)
        np.save(os.path.join(out_unct_v_dir, out_name), unct_v)

### 参考efficient-uncert··· validate.py
def output_pred_unctwithdecom(args,experiment_dir,out_name, label, pred, unct_r, unct_e,unct_b_ue,unct_b_ua,unct_v ):
    out_label_dir = os.path.join(experiment_dir, 'output_val', 'label')
    out_pred_dir = os.path.join(experiment_dir, 'output_val', 'pred')
    #out_preds_dir = os.path.join(experiment_dir, 'output_val', 'preds')
    out_unct_r_dir = os.path.join(experiment_dir, 'output_val', 'unct_r')
    out_unct_e_dir = os.path.join(experiment_dir, 'output_val', 'unct_e')
    out_unct_bue_dir = os.path.join(experiment_dir, 'output_val', 'unct_b_ue')
    out_unct_bua_dir = os.path.join(experiment_dir, 'output_val', 'unct_b_ua')
    out_unct_v_dir = os.path.join(experiment_dir, 'output_val', 'unct_v')
    if not os.path.exists(out_label_dir):
        os.makedirs(out_label_dir)
    if not os.path.exists(out_pred_dir):
        os.makedirs(out_pred_dir)
    '''
    if not os.path.exists(out_preds_dir):
        os.makedirs(out_preds_dir)
    '''
    if not os.path.exists(out_unct_r_dir):
        os.makedirs(out_unct_r_dir)
    if not os.path.exists(out_unct_e_dir):
        os.makedirs(out_unct_e_dir)
    if not os.path.exists(out_unct_bue_dir):
        os.makedirs(out_unct_bue_dir)
    if not os.path.exists(out_unct_bua_dir):
        os.makedirs(out_unct_bua_dir)
    if not os.path.exists(out_unct_v_dir):
        os.makedirs(out_unct_v_dir)

    if args.save_output:
        print("----"+out_name+': outputing'+"----")
        np.save(os.path.join(out_label_dir, out_name), label)
        np.save(os.path.join(out_pred_dir, out_name), pred)
        #np.save(os.path.join(out_preds_dir, out_name), preds)
        np.save(os.path.join(out_unct_r_dir, out_name), unct_r)
        np.save(os.path.join(out_unct_e_dir, out_name), unct_e)
        np.save(os.path.join(out_unct_bue_dir, out_name), unct_b_ue)
        np.save(os.path.join(out_unct_bua_dir, out_name), unct_b_ua)
        np.save(os.path.join(out_unct_v_dir, out_name), unct_v)

#pure estimation
def acquisition_func(acqu, output_mean, square_mean=None, entropy_mean=None):
    if acqu == 'e1':  # max entropy
        return -(output_mean * torch.log(output_mean)).mean(2)
    if acqu == 'e':  # max entropy
        return -(output_mean * torch.log(output_mean)).mean(1)
    elif acqu == 'b':# BALD
        return acquisition_func('e', output_mean) - entropy_mean
    elif acqu == 'r':  # variation ratios
        return 1 - output_mean.max(1)[0]
    elif acqu == 'v':  # mean STD
        return (square_mean - output_mean.pow(2)).mean(1)

#estimation with decomposition
def acquisition_decompos_func(acqu, output_mean, square_mean=None, entropy_mean=None):
    if acqu == 'e1':  # max entropy
        return -(output_mean * torch.log(output_mean)).mean(2)
    if acqu == 'e':  # max entropy
        return -(output_mean * torch.log(output_mean)).mean(1)
    elif acqu == 'b': # BALD
        return acquisition_func('e', output_mean) - entropy_mean, entropy_mean
    elif acqu == 'r':  # variation ratios
        return 1 - output_mean.max(1)[0]
    elif acqu == 'v':  # mean STD
        return (square_mean - output_mean.pow(2)).mean(1)

def log_string(str):
    logger.info(str)
    print(str)

def test_withunct_mcdp(args,classifier, points, scene_label, NUM_CLASSES):
    batchsize = points.size()[0]
    n_pts = points.size()[2]
    for k in range(args.times_var):
        #print("Time:" + str(k))
        classifier = classifier.eval()
        seg_pred, trans_feat = classifier(points)
        seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
        seg_pred = F.softmax(seg_pred, dim=-1)
        if k == 0:
            output_mean = seg_pred * 0  # for pred_label
            output_square = seg_pred * 0  # for unct_square
            entropy_mean = seg_pred.mean(1) * 0  # for unct_entropy
        output_mean += seg_pred
        output_square += seg_pred.pow(2)
        entropy_mean += acquisition_func('e', seg_pred)

    output_mean = output_mean / args.times_var
    output_square = output_square / args.times_var
    entropy_mean = entropy_mean / args.times_var

    pred = torch.max(output_mean, 1)[1].cpu().data.numpy()
    label = scene_label.contiguous().view(-1, 1).cpu().data.numpy()
    # Uncertainty estimation
    unc_map_r = acquisition_func('r', output_mean, \
                                 square_mean=output_square, entropy_mean=entropy_mean)
    unc_map_e = acquisition_func('e', output_mean, \
                                 square_mean=output_square, entropy_mean=entropy_mean)
    unc_map_b = acquisition_func('b', output_mean, \
                                 square_mean=output_square, entropy_mean=entropy_mean)
    unc_map_v = acquisition_func('v', output_mean, \
                                 square_mean=output_square, entropy_mean=entropy_mean)
    unct_r = unc_map_r.data.cpu().numpy()
    unct_e = unc_map_e.data.cpu().numpy()
    unct_b = unc_map_b.data.cpu().numpy()
    unct_v = unc_map_v.data.cpu().numpy()
    output_mean = output_mean.view(batchsize, n_pts, NUM_CLASSES)
    return output_mean, trans_feat, pred, label, unct_r, unct_e, unct_v, unct_b

def test_withunct_mydp(args,classifier, points, scene_label, NUM_CLASSES):
    batchsize = points.size()[0]
    n_pts = points.size()[2]
    for k in range(args.times_var):
        classifier = classifier.eval()
        seg_pred, trans_feat = classifier(points)
        seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
        seg_pred = F.softmax(seg_pred, dim=-1)
        if k == 0:
            output_mean = seg_pred * 0  # for pred_label
            output_square = seg_pred * 0  # for unct_square
            entropy_mean = seg_pred.mean(1) * 0  # for unct_entropy
        output_mean += seg_pred
        output_square += seg_pred.pow(2)
        entropy_mean += acquisition_func('e', seg_pred)

    output_mean = output_mean / args.times_var
    output_square = output_square / args.times_var
    entropy_mean = entropy_mean / args.times_var

    pred = torch.max(output_mean, 1)[1].cpu().data.numpy()
    label = scene_label.contiguous().view(-1, 1).cpu().data.numpy()
    # Uncertainty estimation
    unc_map_r = acquisition_func('r', output_mean, \
                                 square_mean=output_square, entropy_mean=entropy_mean)
    unc_map_e = acquisition_func('e', output_mean, \
                                 square_mean=output_square, entropy_mean=entropy_mean)
    unc_map_b = acquisition_func('b', output_mean, \
                                 square_mean=output_square, entropy_mean=entropy_mean)
    unc_map_v = acquisition_func('v', output_mean, \
                                 square_mean=output_square, entropy_mean=entropy_mean)
    unct_r = unc_map_r.data.cpu().numpy()
    unct_e = unc_map_e.data.cpu().numpy()
    unct_b = unc_map_b.data.cpu().numpy()
    unct_v = unc_map_v.data.cpu().numpy()
    output_mean = output_mean.view(batchsize, n_pts, NUM_CLASSES)
    return output_mean, trans_feat, pred, label, unct_r, unct_e, unct_v, unct_b