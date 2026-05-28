import os
import numpy as np
import torch
import warnings
from config import parse_args
from datasets import MVTEC_CLASS_NAMES, BTAD_CLASS_NAMES, VISA_CLASS_NAMES, ISBI2016_CLASS_NAMES
from utils.utils import init_seeds, setting_lr_parameters
import logging
from datetime import datetime


def main_single(args):
    # model path
    args.model_path = "{}_{}_{}_{}".format(
        args.dataset, args.backbone_arch, args.flow_arch, args.class_name)
    
    # image
    args.img_size = (args.inp_size, args.inp_size)  
    args.crop_size = (args.inp_size, args.inp_size)  
    args.norm_mean, args.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    args.img_dims = [3] + list(args.img_size)

    # output settings
    args.save_results = True
    
    # unsup-train lr settings
    setting_lr_parameters(args)
    
    # selecting train functions
    if args.with_fas:
        from engines.bgad_fas_train_engine import train
        img_auc, pix_auc, pix_pro = train(args)
    else:
        from engines.bgad_train_engine import train
        img_auc, pix_auc, pix_pro = train(args)

    return img_auc, pix_auc, pix_pro


def main():
    init_seeds(0)
    args = parse_args()

    # setup logging: create log dir and file
    out_base = os.path.join(args.output_dir, args.exp_name)
    log_dir = args.log_dir if args.log_dir is not None else os.path.join(out_base, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    # default log file
    default_log_file = os.path.join(log_dir, f"{args.exp_name}.log")
    log_file = args.log_file if args.log_file is not None else default_log_file

    # configure logging handlers
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    try:
        fh = logging.FileHandler(log_file)
        handlers.append(fh)
    except Exception:
        # fallback: only stream
        pass
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=handlers,
    )
    logging.info(f"Logging to {log_file}")
    args.log_file = log_file

    # setting cuda 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda")

    img_aucs, pix_aucs, pix_pros = [], [], []
    if args.class_name == 'none':  # default training all classes
        if args.dataset == 'mvtec':
            CLASS_NAMES = MVTEC_CLASS_NAMES 
        elif args.dataset == 'btad':
            CLASS_NAMES = BTAD_CLASS_NAMES 
        elif args.dataset == 'visa':
            CLASS_NAMES = VISA_CLASS_NAMES
        elif args.dataset == 'isbi2016':
            CLASS_NAMES = ISBI2016_CLASS_NAMES
        else:
            raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))
    else:
        CLASS_NAMES = [args.class_name]
    for class_name in CLASS_NAMES:
        args.class_name = class_name
        img_auc, pix_auc, pix_pro = main_single(args)
        img_aucs.append(img_auc)
        pix_aucs.append(pix_auc)
        pix_pros.append(pix_pro)
    for i, class_name in enumerate(CLASS_NAMES):
        logging.info(f'{class_name}: Image-AUC: {img_aucs[i]}, Pixel-AUC: {pix_aucs[i]}, Pixel-PRO: {pix_pros[i]}')
    logging.info('Mean Image-AUC: {}'.format(np.mean(img_aucs)))
    logging.info('Mean Pixel-AUC: {}'.format(np.mean(pix_aucs)))
    logging.info('Mean Pixel-PRO: {}'.format(np.mean(pix_pros)))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()

