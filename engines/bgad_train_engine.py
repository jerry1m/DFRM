import os
import math
import csv
import timm
import torch
import numpy as np
import time
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils import t2np, get_logp, adjust_learning_rate, warmup_learning_rate, save_results, save_weights, load_weights
from datasets import create_data_loader
from models import positionalencoding2d, load_flow_model
from losses import compute_uncertainty_aware_boundary, compute_adaptive_boundary_contrastive_loss_normal, normal_adaptive_weighting, UncertaintyBoundary
from utils.visualizer import plot_visualizing_results
from utils.utils import MetricRecorder, calculate_pro_metric, convert_to_anomaly_scores, evaluate_thresholds
import logging

log_theta = torch.nn.LogSigmoid()


def _extract_ub_layer_state(ub, class_key):
    key = str(class_key)
    if not hasattr(ub, 'mu') or not hasattr(ub, 'log_var'):
        return None
    if key not in ub.mu or key not in ub.log_var:
        return None
    mu_c = ub.mu[key].detach()
    log_var_c = ub.log_var[key].detach()
    std = torch.exp(0.5 * log_var_c)
    b_n = torch.mean(mu_c) - ub.k * torch.mean(std)
    b_a = b_n - ub.margin_tau
    unc = torch.mean(std)
    w = 1.0 + ub.lambda_w * unc
    return {
        'uncertainty_mean': float(unc.item()),
        'weight_mean': float(w.item()),
        'ub_b_n': float(b_n.item()),
        'ub_b_a': float(b_a.item()),
    }


def _save_ub_stats_csv(args, rows):
    if not rows:
        return
    out_dir = os.path.join(args.output_dir, args.exp_name, 'logs')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'ub_stats_{args.class_name}.csv')
    fieldnames = [
        'epoch',
        'layer',
        'train_loss',
        'normal_violate_ratio',
        'anomaly_violate_ratio',
        'uncertainty_mean',
        'weight_mean',
        'ub_b_n',
        'ub_b_a',
        'flow_b_n',
        'flow_b_a',
    ]
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_meta_epoch(args, epoch, data_loader, encoder, decoders, optimizer, ubs=None):
    N_batch = 4096
    decoders = [decoder.train() for decoder in decoders]  # 3
    adjust_learning_rate(args, optimizer, epoch)
    I = len(data_loader)
    layer_stats = {
        l: {
            'normal_violate_sum': 0.0,
            'normal_count': 0,
            'anomaly_violate_sum': 0.0,
            'anomaly_count': 0,
            'flow_b_n_sum': 0.0,
            'flow_b_a_sum': 0.0,
            'flow_b_count': 0,
        } for l in range(args.feature_levels)
    }

    for sub_epoch in range(args.sub_epochs):
        # update curriculum-scheduled margin_tau (τ(t)) at the beginning of each sub-epoch
        if ubs is not None:
            try:
                T = float(args.meta_epochs)
                t = float(epoch) + float(sub_epoch) / float(args.sub_epochs)
                tau_min = getattr(args, 'tau_min', 0.02)
                tau_max = getattr(args, 'tau_max', getattr(args, 'margin_tau', 0.1))
                lam = getattr(args, 'tau_lambda', 5.0)
                new_tau = tau_min + (tau_max - tau_min) * math.exp(-lam * t / T)
                for ub in ubs:
                    try:
                        ub.set_margin_tau(new_tau)
                    except Exception:
                        # fallback when method not available
                        ub.margin_tau = float(new_tau)
                logging.getLogger(__name__).info('Set margin_tau to {:.4f} at progress t={:.3f}/{:.1f}'.format(new_tau, t, T))
            except Exception:
                pass

        total_loss, loss_count = 0.0, 0
        for i, (data) in enumerate(tqdm(data_loader)):
            # warm-up learning rate
            lr = warmup_learning_rate(args, epoch, i+sub_epoch*I, I*args.sub_epochs, optimizer)

            image, _, mask, _, _ = data
            image = image.to(args.device)  
            mask = mask.to(args.device)
            with torch.no_grad():
                features = encoder(image)
            for l in range(args.feature_levels):
                e = features[l].detach()  
                bs, dim, h, w = e.size()
                mask_ = F.interpolate(mask, size=(h, w), mode='nearest').squeeze(1)
                mask_ = mask_.reshape(-1)
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                
                # (bs, 128, h, w)
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
                    
                    # first epoch only training normal samples without boundaries
                    if epoch == 0:
                        logps = get_logp(dim, z, log_jac_det) 
                        logps = logps / dim
                        loss = -log_theta(logps).mean()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        loss_count += 1
                    else:
                        logps = get_logp(dim, z, log_jac_det)  
                        logps = logps / dim 
                        if args.focal_weighting:
                            logps_detach = logps.detach()
                            nor_weights = normal_adaptive_weighting(logps_detach)
                            loss_ml = -log_theta(logps) * nor_weights # (256, )
                            loss_ml = torch.mean(loss_ml)
                        else:
                            loss_ml = -log_theta(logps)
                            loss_ml = torch.mean(loss_ml)
            
                        boundaries = compute_uncertainty_aware_boundary(logps, m_b, args.pos_beta, args.margin_tau, args.normalizer)
                        logps_n = logps / args.normalizer
                        b_n_flow = boundaries[0]
                        b_a_flow = boundaries[1]
                        n_mask = (m_b == 0)
                        a_mask = (m_b == 1)
                        if n_mask.any():
                            n_vals = logps_n[n_mask]
                            n_violate = (n_vals <= b_n_flow).float().sum().item()
                            layer_stats[l]['normal_violate_sum'] += float(n_violate)
                            layer_stats[l]['normal_count'] += int(n_vals.numel())
                        if a_mask.any():
                            a_vals = logps_n[a_mask]
                            a_violate = (a_vals >= b_a_flow).float().sum().item()
                            layer_stats[l]['anomaly_violate_sum'] += float(a_violate)
                            layer_stats[l]['anomaly_count'] += int(a_vals.numel())
                        layer_stats[l]['flow_b_n_sum'] += float(b_n_flow.item())
                        layer_stats[l]['flow_b_a_sum'] += float(b_a_flow.item())
                        layer_stats[l]['flow_b_count'] += 1
                        #print('feature level: {}, pos_beta: {}, boudaris: {}'.format(l, args.pos_beta, boundaries))
                        if args.focal_weighting:
                            loss_n_con = compute_adaptive_boundary_contrastive_loss_normal(logps, m_b, boundaries, args.normalizer, weights=nor_weights)
                        else:
                            loss_n_con = compute_adaptive_boundary_contrastive_loss_normal(logps, m_b, boundaries, args.normalizer)
                    
                        # UncertaintyBoundary loss (per-feature-level)
                        ub_loss = torch.tensor(0.0, device=e_b.device, dtype=e_b.dtype)
                        if ubs is not None:
                            try:
                                ub = ubs[l]
                                # categories: use class_name as dataset-level key for all samples
                                categories = [args.class_name] * e_b.size(0)
                                ub_loss = ub(e_b, m_b, categories)
                            except Exception:
                                ub_loss = torch.tensor(0.0, device=e_b.device, dtype=e_b.dtype)

                        loss = loss_ml + loss_n_con + args.ub_weight * ub_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # momentum update of boundary params to stabilize mu/log_var
                        try:
                            if ubs is not None:
                                ub = ubs[l]
                                momentum_val = getattr(args, 'ub_momentum', 0.99)
                                # update using current batch normal features (e_b) and mask m_b
                                try:
                                    ub.update_boundary_momentum(e_b, m_b, momentum=momentum_val, key=args.class_name)
                                except Exception:
                                    # fallback without key
                                    ub.update_boundary_momentum(e_b, m_b, momentum=momentum_val)
                        except Exception:
                            pass
                        loss_item = loss.item()
                        if math.isnan(loss_item):
                            total_loss += 0.0
                            loss_count += 0
                        else:
                            total_loss += loss.item()
                            loss_count += 1

    mean_loss = total_loss / loss_count
    logging.getLogger(__name__).info('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_loss, lr))

    ub_rows = []
    for l in range(args.feature_levels):
        st = layer_stats[l]
        n_ratio = st['normal_violate_sum'] / max(st['normal_count'], 1)
        a_ratio = st['anomaly_violate_sum'] / max(st['anomaly_count'], 1)
        flow_b_n = st['flow_b_n_sum'] / max(st['flow_b_count'], 1)
        flow_b_a = st['flow_b_a_sum'] / max(st['flow_b_count'], 1)

        ub_state = None
        if ubs is not None:
            try:
                ub_state = _extract_ub_layer_state(ubs[l], args.class_name)
            except Exception:
                ub_state = None
        if ub_state is None:
            ub_state = {
                'uncertainty_mean': float('nan'),
                'weight_mean': float('nan'),
                'ub_b_n': float('nan'),
                'ub_b_a': float('nan'),
            }

        ub_rows.append({
            'epoch': int(epoch),
            'layer': int(l + 1),
            'train_loss': float(mean_loss),
            'normal_violate_ratio': float(n_ratio),
            'anomaly_violate_ratio': float(a_ratio),
            'uncertainty_mean': float(ub_state['uncertainty_mean']),
            'weight_mean': float(ub_state['weight_mean']),
            'ub_b_n': float(ub_state['ub_b_n']),
            'ub_b_a': float(ub_state['ub_b_a']),
            'flow_b_n': float(flow_b_n),
            'flow_b_a': float(flow_b_a),
        })

    return ub_rows


def validate(args, epoch, data_loader, encoder, decoders):
    logger = logging.getLogger(__name__)
    logger.info('\nCompute loss and scores on category: {}'.format(args.class_name))
    
    decoders = [decoder.eval() for decoder in decoders]
    
    image_list, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
    logps_list = [list() for _ in range(args.feature_levels)]
    latencies = []
    total_loss, loss_count = 0.0, 0
    with torch.no_grad():
        for i, (image, label, mask, file_name, img_type) in enumerate(tqdm(data_loader)):
            if args.vis:
                image_list.extend(t2np(image))
                file_names.extend(file_name)
                img_types.extend(img_type)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            
            image = image.to(args.device) # single scale
            measure = args.measure_inference
            if measure and torch.cuda.is_available():
                torch.cuda.synchronize(args.device)
            if measure:
                start_time = time.perf_counter()

            features = encoder(image)  # BxCxHxW
            for l in range(args.feature_levels):
                e = features[l]  # BxCxHxW
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
               
                # (bs, 128, h, w)
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
        pix_pro = 0.0
        logger.warning('Skip pixel-level metrics for class %s (img_level=%s, mask_unique=%s).',
                       args.class_name, bool(getattr(args, 'img_level', False)), mask_unique.tolist())
    else:
        pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pix_pro = -1
        if args.pro:
            pix_pro = calculate_pro_metric(scores, gt_mask)
    
    if args.vis and epoch == args.meta_epochs - 1 and (not skip_pixel_metrics):
        img_threshold, pix_threshold = evaluate_thresholds(gt_label, gt_mask, img_scores, scores)
        save_dir = os.path.join(args.output_dir, args.exp_name, 'vis_results', args.class_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_visualizing_results(image_list, scores, img_scores, gt_mask_list, pix_threshold, 
                                 img_threshold, save_dir, file_names, img_types)

    if args.measure_inference and latencies:
        valid_latencies = latencies[args.latency_skip:] if len(latencies) > args.latency_skip else latencies
        valid_latencies = np.asarray(valid_latencies, dtype=np.float64)
        if valid_latencies.size:
            mean_latency = float(np.mean(valid_latencies))
            p90_latency = float(np.percentile(valid_latencies, 90))
            throughput = float(1.0 / mean_latency) if mean_latency > 0 else float('inf')
            logger.info(
                'Inference latency per image: mean {:.2f} ms, p90 {:.2f} ms, throughput {:.2f} img/s ({} batches)'.format(
                    mean_latency * 1000.0, p90_latency * 1000.0, throughput, len(valid_latencies)))

    return img_auc, pix_auc, pix_pro


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
                    out_indices=[i+1 for i in range(args.feature_levels)], pretrained=True)
        encoder = encoder.to(args.device).eval()
    feat_dims = encoder.feature_info.channels()
    
    # Normalizing Flows
    decoders = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    decoders = [decoder.to(args.device) for decoder in decoders]
    params = list(decoders[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(decoders[l].parameters())
    # UncertaintyBoundary modules (one per feature level)
    # initialize with tau_max (initial margin); it will be adapted during training
    init_tau = getattr(args, 'tau_max', getattr(args, 'margin_tau', 0.1))
    ubs = [UncertaintyBoundary(dim, k=2.0, lambda_w=0.1, margin_tau=init_tau, max_grad_clip=args.ub_max_grad_clip).to(args.device) for dim in feat_dims]
    # optimizer (include UncertaintyBoundary params)
    for ub in ubs:
        params += list(ub.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # data loaders
    train_loader, test_loader = create_data_loader(args)
    # stats
    img_auc_obs = MetricRecorder('IMG_AUROC')
    pix_auc_obs = MetricRecorder('PIX_AUROC')
    pix_pro_obs = MetricRecorder('PIX_AUPRO')
    ub_stats_rows = []
    for epoch in range(args.meta_epochs):
        if args.checkpoint:
            load_weights(encoder, decoders, args.checkpoint)

        print('Train meta epoch: {}'.format(epoch))
        epoch_rows = train_meta_epoch(args, epoch, train_loader, encoder, decoders, optimizer, ubs=ubs)
        if epoch_rows:
            ub_stats_rows.extend(epoch_rows)

        img_auc, pix_auc, pix_pro = validate(args, epoch, test_loader, encoder, decoders)

        img_auc_obs.update(100.0 * img_auc, epoch)
        pix_auc_obs.update(100.0 * pix_auc, epoch)
        pix_pro_obs.update(100.0 * pix_pro, epoch)
        
    if args.save_results:
        save_results(img_auc_obs, pix_auc_obs, pix_pro_obs, args.output_dir, args.exp_name, args.model_path, args.class_name)
        save_weights(encoder, decoders, args.output_dir, args.exp_name, args.model_path)  # avoid unnecessary saves
        _save_ub_stats_csv(args, ub_stats_rows)

    return img_auc_obs.max_score, pix_auc_obs.max_score, pix_pro_obs.max_score