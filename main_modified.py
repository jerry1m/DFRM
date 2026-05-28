import os
import numpy as np
import torch
import warnings
from config import parse_args
from datasets import MVTEC_CLASS_NAMES, BTAD_CLASS_NAMES, VISA_CLASS_NAMES, ISBI2016_CLASS_NAMES
from utils.utils import init_seeds, setting_lr_parameters
import logging
from datetime import datetime


def log_run_args(args):
    """Log scalar and simple list/tuple arguments for reproducibility."""
    safe_types = (int, float, str, bool)
    def _is_simple(v):
        if isinstance(v, safe_types):
            return True
        if isinstance(v, (list, tuple)):
            return all(isinstance(i, safe_types) for i in v)
        return False
    filtered = {k: v for k, v in vars(args).items() if _is_simple(v)}
    logging.info('Run arguments: %s', filtered)


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
    
    # import segmentation head and edge detector
    from models.modules import UNetSegmentationHead, EdgeDetector
    
    # initialize segmentation head and edge detector
    args.segmentation_head = None
    args.edge_detector = None
    
    # selecting train functions
    if args.with_fas:
        from engines.bgad_fas_train_engine import train
        
        # get feature dimension info
        import timm
        if args.backbone_arch == 'quaternion_cnn':
            from models.quaternion_backbone import QuaternionBackbone
            encoder = QuaternionBackbone(out_indices=[i+1 for i in range(args.feature_levels)], disable_dfrm=args.disable_dfrm)
            encoder = encoder.to(args.device).eval()
        elif args.backbone_arch == 'resnet_se':
            from models.resnet_se_backbone import ResNetDFRMBackbone
            base_arch = getattr(args, 'backbone_base', 'resnet50')
            encoder = ResNetDFRMBackbone(base_arch=base_arch, out_indices=[i+1 for i in range(args.feature_levels)], pretrained=True, disable_dfrm=args.disable_dfrm)
            encoder = encoder.to(args.device).eval()
        else:
            encoder = timm.create_model(args.backbone_arch, features_only=True, 
                                       out_indices=[i+1 for i in range(args.feature_levels)], 
                                       pretrained=True)
            encoder = encoder.to(args.device).eval()
        feat_dims = encoder.feature_info.channels()
        last_feat_dim = feat_dims[-1]
        
        # create segmentation head and edge detector
        segmentation_head = UNetSegmentationHead(last_feat_dim)
        edge_detector = EdgeDetector()
        
        segmentation_head = segmentation_head.to(args.device)
        edge_detector = edge_detector.to(args.device)
        
        args.segmentation_head = segmentation_head
        args.edge_detector = edge_detector
        
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

    # log training/inference configuration
    log_run_args(args)

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
    
    # add feature levels parameter (if not defined)
    if not hasattr(args, 'feature_levels'):
        args.feature_levels = 3
    
    for class_name in CLASS_NAMES:
        args.class_name = class_name
        img_auc, pix_auc, pix_pro = main_single(args)
        img_aucs.append(img_auc)
        pix_aucs.append(pix_auc)
        pix_pros.append(pix_pro)
    
    # print results with mask AUC
    for i, class_name in enumerate(CLASS_NAMES):
        logging.info(f'{class_name}: Image-AUC: {img_aucs[i]}, Pixel-AUC: {pix_aucs[i]}, Pixel-PRO: {pix_pros[i]}')
    
    # also print mask AUC results if available
    if hasattr(args, 'mask_aucs') and args.mask_aucs:
        for i, class_name in enumerate(CLASS_NAMES):
            logging.info(f'{class_name}: Mask-AUC: {args.mask_aucs[i]}')
        logging.info('Mean Mask-AUC: {}'.format(np.mean(args.mask_aucs)))
    
    logging.info('Mean Image-AUC: {}'.format(np.mean(img_aucs)))
    logging.info('Mean Pixel-AUC: {}'.format(np.mean(pix_aucs)))
    logging.info('Mean Pixel-PRO: {}'.format(np.mean(pix_pros)))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()