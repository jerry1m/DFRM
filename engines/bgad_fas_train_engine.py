import os
import math
import timm
import torch
import numpy as np
import time
import logging
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from utils import t2np, get_logp, adjust_learning_rate, warmup_learning_rate, save_results, save_weights, load_weights
from datasets import create_fas_data_loader
from models import positionalencoding2d, load_flow_model
from losses import compute_uncertainty_aware_boundary, compute_adaptive_boundary_contrastive_loss, normal_adaptive_weighting, abnormal_adaptive_weighting, DiceLoss, BCEWithLogitsLoss
from utils.visualizer import plot_visualizing_results, generate_high_quality_mask, generate_precise_edges, extract_filename_info
from utils.utils import MetricRecorder, calculate_pro_metric, convert_to_anomaly_scores, evaluate_thresholds

log_theta = torch.nn.LogSigmoid()
logger = logging.getLogger(__name__)

def train_meta_epoch(args, epoch, data_loader, encoder, decoders, optimizer, segmentation_head, edge_detector):
    N_batch = 4096
    decoders = [decoder.train() for decoder in decoders]  # 3
    
    # add segmentation loss function
    seg_criterion = DiceLoss()
    edge_criterion = BCEWithLogitsLoss()
    
    adjust_learning_rate(args, optimizer, epoch)
    I = len(data_loader)
    
    # First epoch only training on normal samples to keep training steadily
    if epoch == 0:
        data_loader = data_loader[0]
    else:
        data_loader = data_loader[1]
    
    for sub_epoch in range(args.sub_epochs):
        total_loss, loss_count = 0.0, 0
        for i, (data) in enumerate(tqdm(data_loader)):
            # warm-up learning rate
            lr = warmup_learning_rate(args, epoch, i+sub_epoch*I, I*args.sub_epochs, optimizer)
            
            if epoch == 0:
                image, _, mask, _, _ = data
            else:
                image, _, mask = data
            
            image = image.to(args.device)
            mask = mask.to(args.device)
            
            with torch.no_grad():
                features = encoder(image)
            
            # last layer features for segmentation (move forward to sub-batch to avoid reusing same computation graph in backward)
            seg_features = features[-1]  # use the last feature layer
            
            for l in range(args.feature_levels):
                e = features[l].detach()
                bs, dim, h, w = e.size()
                
                mask_ = F.interpolate(mask, size=(h, w), mode='nearest').squeeze(1)
                mask_ = mask_.reshape(-1)
                
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)  # (bs, 128, h, w)
                
                pos_embed = positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                
                decoder = decoders[l]
                
                perm = torch.randperm(bs*h*w).to(args.device)
                num_N_batches = bs*h*w // N_batch
                
                for i in range(num_N_batches):
                    idx = torch.arange(i*N_batch, (i+1)*N_batch)
                    p_b = pos_embed[perm[idx]]
                    e_b = e[perm[idx]]
                    m_b = mask_[perm[idx]]
                    
                    if args.flow_arch == 'flow_model':
                        z, log_jac_det = decoder(e_b)
                    else:
                        z, log_jac_det = decoder(e_b, [p_b, ])
                    
                    # first epoch only training normal samples
                    if epoch == 0:
                        if m_b.sum() == 0:  # only normal loss
                            logps = get_logp(dim, z, log_jac_det)
                            logps = logps / dim
                            
                            loss = -log_theta(logps).mean()
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            
                            total_loss += loss.item()
                            loss_count += 1
                    
                    else:
                        if m_b.sum() == 0:  # only normal ml loss
                            logps = get_logp(dim, z, log_jac_det)
                            logps = logps / dim
                            
                            if args.focal_weighting:
                                normal_weights = normal_adaptive_weighting(logps.detach())
                                loss = -log_theta(logps) * normal_weights
                                loss = loss.mean()
                            else:
                                loss = -log_theta(logps).mean()
                            
                        if m_b.sum() > 0:  # normal ml loss and bg_sppc loss
                            logps = get_logp(dim, z, log_jac_det)
                            logps = logps / dim
                            
                            if args.focal_weighting:
                                logps_detach = logps.detach()
                                normal_logps = logps_detach[m_b == 0]
                                anomaly_logps = logps_detach[m_b == 1]
                                
                                nor_weights = normal_adaptive_weighting(normal_logps)
                                ano_weights = abnormal_adaptive_weighting(anomaly_logps)
                                
                                weights = nor_weights.new_zeros(logps_detach.shape)
                                weights[m_b == 0] = nor_weights
                                weights[m_b == 1] = ano_weights
                                
                                loss_ml = -log_theta(logps[m_b == 0]) * nor_weights  # (256, )
                                loss_ml = torch.mean(loss_ml)
                            else:
                                loss_ml = -log_theta(logps[m_b == 0])
                                loss_ml = torch.mean(loss_ml)
                            
                            boundaries = compute_uncertainty_aware_boundary(logps, m_b, args.pos_beta, args.margin_tau, args.normalizer)
                            
                            if args.focal_weighting:
                                loss_n_con, loss_a_con = compute_adaptive_boundary_contrastive_loss(logps, m_b, boundaries, args.normalizer, weights=weights)
                            else:
                                loss_n_con, loss_a_con = compute_adaptive_boundary_contrastive_loss(logps, m_b, boundaries, args.normalizer)
                            
                            loss = loss_ml + args.bgspp_lambda * (loss_n_con + loss_a_con)
                        
                        # recompute segmentation and edges within each sub-batch to build independent computation graph, avoid double backward error
                        defect_mask = segmentation_head(seg_features)
                        edges = edge_detector(defect_mask)
                        target_mask = F.interpolate(mask.float(), size=defect_mask.shape[2:], mode='bilinear')
                        target_edges = edge_detector(target_mask)

                        # add segmentation loss
                        seg_loss = seg_criterion(defect_mask, target_mask)
                        edge_loss = edge_criterion(edges, target_edges)
                        
                        # combined loss
                        total_loss_val = loss + 0.3 * seg_loss + 0.15 * edge_loss
                        
                        optimizer.zero_grad()
                        total_loss_val.backward()
                        optimizer.step()
                        
                        loss_item = total_loss_val.item()
                        if math.isnan(loss_item):
                            total_loss += 0.0
                            loss_count += 0
                        else:
                            total_loss += loss_item
                            loss_count += 1
    
    if loss_count > 0:
        mean_loss = total_loss / loss_count
        # Print epoch and sub-epoch explicitly to avoid confusion like '0.7'
        logger.info('Epoch: {} Sub-epoch: {} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_loss, lr))

def validate(args, epoch, data_loader, encoder, decoders, segmentation_head, edge_detector):
    logger.info('\nCompute loss and scores on category: {}'.format(args.class_name))
    decoders = [decoder.eval() for decoder in decoders]
    
    image_list, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
    logps_list = [list() for _ in range(args.feature_levels)]
    all_pred_masks = []
    all_pred_edges = []
    latencies = []
    
    total_loss, loss_count = 0.0, 0
    
    with torch.no_grad():
        for i, (image, label, mask, file_name, img_type) in enumerate(tqdm(data_loader)):
            # image: (32, 3, 256); label: (32, ); mask: (32, 1, 256, 256)
            # Always collect ground-truth labels and masks for metric computation.
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))

            if args.vis:
                image_list.extend(t2np(image))
                file_names.extend(file_name)
                img_types.extend(img_type)
            
            image = image.to(args.device)
            measure = args.measure_inference
            if measure and torch.cuda.is_available():
                torch.cuda.synchronize(args.device)
            if measure:
                start_time = time.perf_counter()
            
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
                
                loss = -log_theta(logps).mean()
                total_loss += loss.item()
                loss_count += 1
                
                logps_list[l].append(logps.reshape(bs, h, w))

            if measure:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(args.device)
                elapsed = time.perf_counter() - start_time
                latencies.append(elapsed / image.size(0))
    
    mean_loss = total_loss / loss_count
    logger.info('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, mean_loss))
    
    scores = convert_to_anomaly_scores(args, logps_list)
    
    # calculate detection AUROC
    img_scores = np.max(scores, axis=(1, 2))
    gt_label = np.asarray(gt_label_list, dtype=bool)
    img_auc = roc_auc_score(gt_label, img_scores)
    
    # calculate segmentation AUROC (optional in image-level-only mode)
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
    mask_unique = np.unique(gt_mask)
    has_valid_pixel_gt = mask_unique.size > 1
    skip_pixel_metrics = bool(getattr(args, 'img_level', False)) or (not has_valid_pixel_gt)
    if skip_pixel_metrics:
        pix_auc = 0.0
        logger.warning('Skip pixel/mask metrics for class %s (img_level=%s, mask_unique=%s).',
                       args.class_name, bool(getattr(args, 'img_level', False)), mask_unique.tolist())
    else:
        pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    
    # calculate mask AUROC
    # all_pred_masks: list of batch numpy arrays (B, C, h, w) -- convert to per-image list
    pred_list = []
    for batch_pred in all_pred_masks:
        bp = np.asarray(batch_pred)
        if bp.ndim == 4:
            for i in range(bp.shape[0]):
                pred_list.append(bp[i])
        elif bp.ndim == 3:
            for i in range(bp.shape[0]):
                pred_list.append(bp[i])

    # If lengths mismatch, warn and try to proceed with the min length
    if len(pred_list) != len(gt_mask_list):
        logger.warning('Warning: number of predicted masks (%d) != number of gt masks (%d). Using pairwise min length.', len(pred_list), len(gt_mask_list))

    pair_len = min(len(pred_list), len(gt_mask_list))

    flat_pred_masks_list = []
    flat_gt_masks_list = []

    for i in range(pair_len):
        pred = np.asarray(pred_list[i]).squeeze()
        gt = np.asarray(gt_mask_list[i]).squeeze()

        # ensure pred and gt are 2D
        if pred.ndim == 3:
            pred = pred[0]
        if gt.ndim == 3:
            gt = gt[0]

        # upsample pred to gt size using torch interpolate for reliability
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float().to(args.device)
        target_h, target_w = gt.shape[-2], gt.shape[-1]
        pred_up = F.interpolate(pred_t, size=(target_h, target_w), mode='bilinear', align_corners=False)
        pred_up = pred_up.squeeze().cpu().numpy()

        flat_pred_masks_list.append(pred_up.ravel())
        flat_gt_masks_list.append(gt.ravel())

    if skip_pixel_metrics:
        mask_auc = 0.0
    elif len(flat_pred_masks_list) == 0:
        logger.warning('No mask predictions available to compute mask AUC; setting mask_auc = -1')
        mask_auc = -1
    else:
        flat_pred_masks = np.concatenate(flat_pred_masks_list)
        flat_gt_masks = np.concatenate(flat_gt_masks_list)
        mask_auc = roc_auc_score(flat_gt_masks, flat_pred_masks)
    
    pix_pro = 0.0 if skip_pixel_metrics else -1
    if (not skip_pixel_metrics) and args.pro:
        pix_pro = calculate_pro_metric(scores, gt_mask)
    
    if args.vis and epoch == args.meta_epochs - 1 and (not skip_pixel_metrics):
        img_threshold, pix_threshold = evaluate_thresholds(gt_label, gt_mask, img_scores, scores)
        save_dir = os.path.join(args.output_dir, args.exp_name, 'vis_results', args.class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # save mask and edge results (ensure 2D array and normalize)
        for i in range(len(all_pred_masks)):
            # use extract_filename_info to produce consistent base names (e.g., 'manipulated_front_000')
            fname = file_names[i] if i < len(file_names) else None
            itype = img_types[i] if i < len(img_types) else None
            base_name = extract_filename_info(fname, itype)
            mask_path = os.path.join(save_dir, f'{base_name}_mask.png')
            edge_path = os.path.join(save_dir, f'{base_name}_edge.png')

            pred_mask = np.asarray(all_pred_masks[i])
            pred_edge = np.asarray(all_pred_edges[i])

            # take the first prediction in batch (validate appends one batch at a time)
            if pred_mask.ndim == 4:
                m = pred_mask[0]
            elif pred_mask.ndim == 3:
                m = pred_mask[0]
            else:
                m = pred_mask

            if pred_edge.ndim == 4:
                e = pred_edge[0]
            elif pred_edge.ndim == 3:
                e = pred_edge[0]
            else:
                e = pred_edge

            # remove channel axis, ensure (H, W)
            m = np.squeeze(m)
            e = np.squeeze(e)

            # normalize to [0,1], avoid issues with constant arrays
            try:
                if m.max() != m.min():
                    m = (m - m.min()) / (m.max() - m.min())
            except Exception:
                pass
            try:
                if e.max() != e.min():
                    e = (e - e.min()) / (e.max() - e.min())
            except Exception:
                pass

            plt.imsave(mask_path, m, cmap='gray')
            plt.imsave(edge_path, e, cmap='gray')
        
        plot_visualizing_results(image_list, scores, img_scores, gt_mask_list, pix_threshold, img_threshold, save_dir, file_names, img_types)
    
    if args.measure_inference and latencies:
        valid_latencies = latencies[args.latency_skip:] if len(latencies) > args.latency_skip else latencies
        valid_latencies = np.asarray(valid_latencies, dtype=np.float64)
        if valid_latencies.size:
            mean_latency = float(np.mean(valid_latencies))
            p90_latency = float(np.percentile(valid_latencies, 90))
            throughput = float(1.0 / mean_latency) if mean_latency > 0 else float('inf')
            logger.info('Inference latency per image: mean {:.2f} ms, p90 {:.2f} ms, throughput {:.2f} img/s ({} batches)'.format(
                mean_latency * 1000.0, p90_latency * 1000.0, throughput, len(valid_latencies)))
    
    return img_auc, pix_auc, pix_pro, mask_auc

def train(args):
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

    # get segmentation head and edge detector passed from args
    segmentation_head = args.segmentation_head
    edge_detector = args.edge_detector
    
    # optimizer - add segmentation head and edge detector parameters
    params = list(decoders[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(decoders[l].parameters())
    params += list(segmentation_head.parameters())
    params += list(edge_detector.parameters())
    
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # data loaders
    normal_loader, train_loader, test_loader = create_fas_data_loader(args)
    
    # stats
    img_auc_obs = MetricRecorder('IMG_AUROC')
    pix_auc_obs = MetricRecorder('PIX_AUROC')
    pix_pro_obs = MetricRecorder('PIX_AUPRO')
    mask_auc_obs = MetricRecorder('MASK_AUROC')
    
    # load checkpoint (if any)
    if args.checkpoint:
        load_weights(encoder, decoders, args.anomaly_source_path)
        # load segmentation head and edge detector weights
        if hasattr(args, 'segmentation_head_path') and args.segmentation_head_path:
            segmentation_head.load_state_dict(torch.load(args.segmentation_head_path))
        if hasattr(args, 'edge_detector_path') and args.edge_detector_path:
            edge_detector.load_state_dict(torch.load(args.edge_detector_path))
    
    # track best mask AUC to save best seg/edge checkpoints
    best_mask_auc = -1.0
    for epoch in range(args.meta_epochs):
        logger.info('Train meta epoch: {}'.format(epoch))
        
        # train - pass segmentation head and edge detector
        train_meta_epoch(args, epoch, [normal_loader, train_loader], encoder, decoders, optimizer,
                        segmentation_head, edge_detector)
        
        # validate - pass segmentation head and edge detector
        img_auc, pix_auc, pix_pro, mask_auc = validate(args, epoch, test_loader, encoder, decoders,
                                                      segmentation_head, edge_detector)

        img_auc_obs.update(100.0 * img_auc, epoch)
        pix_auc_obs.update(100.0 * pix_auc, epoch)
        pix_pro_obs.update(100.0 * pix_pro, epoch)
        mask_auc_obs.update(100.0 * mask_auc, epoch)

        pix_pro_str = '{:.2f}'.format(pix_pro_obs.last) if pix_pro is not None else 'N/A'
        logger.info('Current IMG_AUROC: {:.2f}, PIX_AUROC: {:.2f}, PIX_PRO: {}, MASK_AUROC: {:.2f}'.format(
            img_auc_obs.last, pix_auc_obs.last, pix_pro_str, mask_auc_obs.last))
        
        if args.save_results:
            save_results(img_auc_obs, pix_auc_obs, pix_pro_obs, args.output_dir, args.exp_name, args.model_path, args.class_name)
            # always save flow weights (encoder + decoders)
            save_weights(encoder, decoders, args.output_dir, args.exp_name, args.model_path)

            # Only save segmentation_head and edge_detector when mask AUC improves
            try:
                current_mask_auc = float(mask_auc)
            except Exception:
                current_mask_auc = None

            if current_mask_auc is not None and current_mask_auc > best_mask_auc:
                best_mask_auc = current_mask_auc
                seg_save_path = os.path.join(args.output_dir, args.exp_name, 'segmentation_head_{}.pt'.format(args.class_name))
                edge_save_path = os.path.join(args.output_dir, args.exp_name, 'edge_detector_{}.pt'.format(args.class_name))
                torch.save(segmentation_head.state_dict(), seg_save_path)
                torch.save(edge_detector.state_dict(), edge_save_path)
                logger.info('Saved best segmentation/edge weights (mask_auc={:.4f}) to {} and {}'.format(best_mask_auc, seg_save_path, edge_save_path))
    
    # save mask AUC result to args
    if not hasattr(args, 'mask_aucs'):
        args.mask_aucs = []
    args.mask_aucs.append(mask_auc_obs.max_score)
    
    logger.info('Max IMG_AUROC: {:.2f}, PIX_AUROC: {:.2f}, PIX_PRO: {:.2f}, MASK_AUROC: {:.2f}'.format(
        img_auc_obs.max_score, pix_auc_obs.max_score, pix_pro_obs.max_score, mask_auc_obs.max_score))
    
    return img_auc_obs.max_score, pix_auc_obs.max_score, pix_pro_obs.max_score