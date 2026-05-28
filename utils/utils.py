import numpy as np
import math
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve 
from skimage.measure import label, regionprops


np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))


class MetricRecorder:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, print_score=True):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        import logging
        logger = logging.getLogger(__name__)
        logger.info('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))


class EachEpochRecorder:
    def __init__(self):
        self.img_aucs = []
        self.pix_aucs = []
        self.train_losses = []
        self.test_losses = []
    
    def update(self, img_auc, pix_auc, train_loss, test_loss):
        self.img_aucs.append(img_auc)
        self.pix_aucs.append(pix_auc)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
    
    def write_to_file(self, file_name):
        img_auc_line = ','.join(self.img_aucs)
        pix_auc_line = ','.join(self.pix_aucs)
        train_loss_line = ','.join(self.train_losses)
        test_loss_line = ','.join(self.test_losses)
        with open(file_name, 'w') as f:
            f.write(img_auc_line)
            f.write(pix_auc_line)
            f.write(train_loss_line)
            f.write(test_loss_line)

            
def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J
    return logp


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setting_lr_parameters(args):
    args.scaled_lr_decay_epochs = [i*args.meta_epochs // 100 for i in args.lr_decay_epochs]
    print('LR schedule: {}'.format(args.scaled_lr_decay_epochs))
    if args.lr_warm:
        args.lr_warmup_from = args.lr / 10.0
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.lr_warm_epochs / args.meta_epochs)) / 2
        else:
            args.lr_warmup_to = args.lr


def calculate_pro_metric(scores, gt_mask):
    
    max_step = 1000
    expect_fpr = 0.3  # default 30%
    # guard against degenerate scores
    try:
        max_th = float(scores.max())
        min_th = float(scores.min())
    except Exception:
        return 0.0
    if not np.isfinite(max_th) or not np.isfinite(min_th):
        return 0.0
    if max_th - min_th <= 0:
        return 0.0
    delta = (max_th - min_th) / max_step
    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1
        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map 
        for i in range(len(binary_score_maps)):    # for i th image
            # pro (per region level)
            label_map = label(gt_mask[i], connectivity=2)
            props = regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox    # find the bounding box of an anomaly region 
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = gt_mask[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image    # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], gt_mask[i]).astype(np.float32).sum()
            # when the gt have no anomaly pixels, skip it
            if gt_mask[i].any():
                if union > 0:
                    iou.append(intersection / union)
        # against steps and average metrics on the testing data
        # handle empty lists
        if len(iou) > 0:
            ious_mean.append(np.array(iou).mean())
            ious_std.append(np.array(iou).std())
        else:
            ious_mean.append(0.0)
            ious_std.append(0.0)
        if len(pro) > 0:
            pros_mean.append(np.array(pro).mean())
            pros_std.append(np.array(pro).std())
        else:
            pros_mean.append(0.0)
            pros_std.append(0.0)
        # fpr for pro-auc
        gt_masks_neg = ~gt_mask
        denom = gt_masks_neg.sum()
        if denom == 0:
            fpr = 0.0
        else:
            fpr = np.logical_and(gt_masks_neg, binary_score_maps).sum() / denom
        fprs.append(fpr)
        threds.append(thred)
    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)
    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)
    # best per image iou
    try:
        best_miou = np.max(ious_mean)
    except Exception:
        best_miou = 0.0
    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    pros_mean_selected = np.array(pros_mean)[idx]
    # if nothing selected, return 0.0
    if fprs_selected.size == 0 or pros_mean_selected.size == 0:
        return 0.0
    # rescale fpr [0,0.3] -> [0, 1]
    try:
        fprs_selected = rescale(fprs_selected)
        pix_pro_auc = auc(fprs_selected, pros_mean_selected)
    except Exception:
        pix_pro_auc = 0.0

    return float(pix_pro_auc)


def evaluate_thresholds(gt_label, gt_mask, img_scores, scores):
    gt_label = np.asarray(gt_label).astype(np.int64)
    if np.unique(gt_label).size < 2:
        image_threshold = float(np.median(img_scores)) if np.size(img_scores) else 0.0
    else:
        precision, recall, thresholds = precision_recall_curve(gt_label, img_scores)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        image_threshold = thresholds[np.argmax(f1)]

    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    pixel_threshold = thresholds[np.argmax(f1)]

    return image_threshold, pixel_threshold



def convert_to_anomaly_scores(args, logps_list):
    normal_map = [list() for _ in range(args.feature_levels)]
    for l in range(args.feature_levels):
        logps = torch.cat(logps_list[l], dim=0)  
        logps-= torch.max(logps) # normalize log-likelihoods to (-Inf:0] by subtracting a constant
        probs = torch.exp(logps) # convert to probs in range [0:1]
        # upsample
        normal_map[l] = F.interpolate(probs.unsqueeze(1),
            size=args.img_size, mode='bilinear', align_corners=True).squeeze().cpu().numpy()
    
    # score aggregation
    scores = np.zeros_like(normal_map[0])
    for l in range(args.feature_levels):
        scores += normal_map[l]

    # normality score to anomaly score
    scores = scores.max() - scores 

    return scores


def count_paprameters(model):
    return sum(p.numel() for p in model.parameters())

