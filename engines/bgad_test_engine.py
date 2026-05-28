import os
import math
import timm
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from utils import t2np, get_logp, load_weights
from datasets import create_test_data_loader
from models import positionalencoding2d, load_flow_model
from utils.visualizer import plot_visualizing_results, generate_high_quality_mask, generate_precise_edges, extract_filename_info
from utils.utils import calculate_pro_metric, convert_to_anomaly_scores, evaluate_thresholds


def _stack_mask_list(mask_list):
    if not mask_list:
        return np.array([])
    stacked = np.asarray(mask_list, dtype=bool)
    if stacked.ndim == 4 and stacked.shape[1] == 1:
        stacked = np.squeeze(stacked, axis=1)
    elif stacked.ndim == 3:
        stacked = stacked
    else:
        stacked = np.squeeze(stacked)
    return stacked

def validate(args, data_loader, encoder, decoders, segmentation_head, edge_detector):
    print('\nCompute loss and scores on category: {}'.format(args.class_name))
    decoders = [decoder.eval() for decoder in decoders]
    
    image_list, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
    logps_list = [list() for _ in range(args.feature_levels)]
    all_pred_masks = []
    all_pred_edges = []
    
    with torch.no_grad():
        for i, (image, label, mask, file_name, img_type) in enumerate(tqdm(data_loader)):
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            if args.vis:
                image_list.extend(t2np(image))
                file_names.extend(file_name)
                img_types.extend(img_type)
            
            image = image.to(args.device)
            
            # single scale
            features = encoder(image)  # BxCxHxW
            
            # generate defect mask and edges
            seg_features = features[-1]
            defect_mask = segmentation_head(seg_features)
            edges = edge_detector(defect_mask)
            
            all_pred_masks.append(defect_mask.cpu().numpy())
            all_pred_edges.append(edges.cpu().numpy())
            
            for l in range(args.feature_levels):
                e = features[l]  # BxCxHxW
                bs, dim, h, w = e.size()
                
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)  # (bs, 128, h, w)
                
                pos_embed = positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                
                decoder = decoders[l]
                
                if args.flow_arch == 'flow_model':
                    z, log_jac_det = decoder(e)
                else:
                    z, log_jac_det = decoder(e, [pos_embed, ])
                
                logps = get_logp(dim, z, log_jac_det)
                logps = logps / dim
                
                logps_list[l].append(logps.reshape(bs, h, w))
    
    scores = convert_to_anomaly_scores(args, logps_list)

    # calculate detection AUROC
    img_scores = np.max(scores, axis=(1, 2))
    gt_label = np.asarray(gt_label_list, dtype=bool)
    if gt_label.size and np.unique(gt_label).size > 1:
        img_auc = roc_auc_score(gt_label, img_scores)
    else:
        img_auc = -1

    # calculate segmentation AUROC
    gt_mask = _stack_mask_list(gt_mask_list)
    pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten()) if gt_mask.size else -1

    # calculate mask AUROC (upsample preds to gt size per image)
    flat_pred_masks_list = []
    flat_gt_masks_list = []
    for i in range(len(all_pred_masks)):
        pred = np.asarray(all_pred_masks[i]).squeeze()
        gt = np.asarray(gt_mask_list[i]).squeeze()
        if pred.ndim == 3:
            pred = pred[0]
        if gt.ndim == 3:
            gt = gt[0]
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float().to(args.device)
        target_h, target_w = gt.shape[-2], gt.shape[-1]
        pred_up = F.interpolate(pred_t, size=(target_h, target_w), mode='bilinear', align_corners=False)
        pred_up = pred_up.squeeze().cpu().numpy()
        flat_pred_masks_list.append(pred_up.ravel())
        flat_gt_masks_list.append(gt.ravel())
    if flat_pred_masks_list:
        flat_pred_masks = np.concatenate(flat_pred_masks_list)
        flat_gt_masks = np.concatenate(flat_gt_masks_list)
        mask_auc = roc_auc_score(flat_gt_masks, flat_pred_masks)
    else:
        mask_auc = -1

    pix_pro = -1
    if args.pro:
        pix_pro = calculate_pro_metric(scores, gt_mask)

    if args.vis:
        img_threshold, pix_threshold = evaluate_thresholds(gt_label, gt_mask, img_scores, scores)
        save_dir = os.path.join(args.output_dir, args.exp_name, 'vis_results', args.class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        for i in range(len(all_pred_masks)):
            fname = file_names[i] if i < len(file_names) else None
            itype = img_types[i] if i < len(img_types) else None
            base_name = extract_filename_info(fname, itype)
            mask_path = os.path.join(save_dir, f'{base_name}_mask.png')
            edge_path = os.path.join(save_dir, f'{base_name}_edge.png')

            pred_mask = np.asarray(all_pred_masks[i]).squeeze()
            pred_edge = np.asarray(all_pred_edges[i]).squeeze()
            if pred_mask.ndim == 3:
                pred_mask = pred_mask[0]
            if pred_edge.ndim == 3:
                pred_edge = pred_edge[0]

            try:
                if pred_mask.max() != pred_mask.min():
                    pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
            except Exception:
                pass
            try:
                if pred_edge.max() != pred_edge.min():
                    pred_edge = (pred_edge - pred_edge.min()) / (pred_edge.max() - pred_edge.min())
            except Exception:
                pass

            if plt is not None:
                plt.imsave(mask_path, pred_mask, cmap='gray')
                plt.imsave(edge_path, pred_edge, cmap='gray')
        
        plot_visualizing_results(image_list, scores, img_scores, gt_mask_list, pix_threshold, img_threshold, save_dir, file_names, img_types)
    
    return img_auc, pix_auc, pix_pro, mask_auc

def test(args):
    # Feature Extractor
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
    
    # Normalizing Flows
    decoders = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    decoders = [decoder.to(args.device) for decoder in decoders]
    
    # import segmentation head and edge detector
    from models.modules import UNetSegmentationHead, EdgeDetector
    
    # initialize segmentation head and edge detector
    last_feat_dim = feat_dims[-1]
    segmentation_head = UNetSegmentationHead(last_feat_dim)
    edge_detector = EdgeDetector()
    
    segmentation_head = segmentation_head.to(args.device)
    edge_detector = edge_detector.to(args.device)
    
    # data loaders
    test_loader = create_test_data_loader(args)
    
    checkpoint = os.path.join(
        args.checkpoint,
        f'{args.dataset}_{args.backbone_arch}_{args.flow_arch}_{args.class_name}.pt'
    )
    if args.checkpoint:
        load_weights(encoder, decoders, checkpoint)
        
        # load segmentation head and edge detector weights (prefer class-suffixed)
        seg_candidates = [
            os.path.join(args.checkpoint, f'segmentation_head_{args.class_name}.pt'),
            os.path.join(args.checkpoint, 'segmentation_head.pt')
        ]
        edge_candidates = [
            os.path.join(args.checkpoint, f'edge_detector_{args.class_name}.pt'),
            os.path.join(args.checkpoint, 'edge_detector.pt')
        ]

        for seg_path in seg_candidates:
            if os.path.exists(seg_path):
                segmentation_head.load_state_dict(torch.load(seg_path, map_location=args.device))
                print(f"Loaded segmentation head weights from {seg_path}")
                break

        for edge_path in edge_candidates:
            if os.path.exists(edge_path):
                edge_detector.load_state_dict(torch.load(edge_path, map_location=args.device))
                print(f"Loaded edge detector weights from {edge_path}")
                break
    
    img_auc, pix_auc, pix_pro, mask_auc = validate(args, test_loader, encoder, decoders,
                                                  segmentation_head, edge_detector)
    
    print('{} Image AUC: {:.2f}'.format(args.class_name, img_auc * 100))
    print('{} Pixel AUC: {:.2f}'.format(args.class_name, pix_auc * 100))
    print('{} Pixel PRO: {:.2f}'.format(args.class_name, pix_pro * 100))
    print('{} Mask AUC: {:.2f}'.format(args.class_name, mask_auc * 100))
    
    return img_auc, pix_auc, pix_pro, mask_auc