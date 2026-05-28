import os, math
import numpy as np
import torch

RESULT_DIR = './results'
WEIGHT_DIR = './weights'
MODEL_DIR  = './models'

__all__ = ('save_results', 'save_weights', 'load_weights', 'adjust_learning_rate', 'warmup_learning_rate')

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, output_dir, exp_name, model_path, class_name):
    result = '{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} at epoch {:d}/{:d}/{:d} for {:s}\n'.format(
        det_roc_obs.max_score, seg_roc_obs.max_score, seg_pro_obs.max_score,
        det_roc_obs.name, seg_roc_obs.name, seg_pro_obs.name,
        det_roc_obs.max_epoch, seg_roc_obs.max_epoch, seg_pro_obs.max_epoch, class_name)
    save_dir = os.path.join(output_dir, exp_name, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fp = open(os.path.join(save_dir, '{}.txt'.format(model_path)), "w")
    fp.write(result)
    fp.close()


def save_weights(encoder, decoders, output_dir, exp_name, model_path):
    model_dir = os.path.join(output_dir, exp_name, 'weights')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    state = {'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': [decoder.state_dict() for decoder in decoders] if isinstance(decoders, list) else decoders.state_dict()}
    filename = '{}.pt'.format(model_path)
    torch.save(state, os.path.join(model_dir, filename))
    print('Saving weights to {}'.format(os.path.join(model_dir, filename)))


def load_weights(encoder, decoders, filename):
    #path = os.path.join(WEIGHT_DIR, filename)
    state = torch.load(filename)
    # Safely load encoder weights: only copy parameters whose shapes match.
    enc_state = state.get('encoder_state_dict', {})
    if enc_state:
        model_sd = encoder.state_dict()
        loaded = 0
        for k, v in enc_state.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
        encoder.load_state_dict(model_sd)
        print(f'Loaded {loaded} encoder params from {filename} (shape-matching)')
    else:
        print('No encoder_state_dict found in checkpoint')

    # Safely load decoder weights: for each decoder, only copy matching-shape params.
    dec_states = state.get('decoder_state_dict', None)
    if dec_states is None:
        print('No decoder_state_dict found in checkpoint')
    else:
        used_idxs = set()
        for idx, decoder in enumerate(decoders):
            model_sd = decoder.state_dict()
            # try to determine model's expected channel dim from a representative param
            rep_dim = None
            for k, v in model_sd.items():
                if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] == 1:
                    rep_dim = v.shape[1]
                    break
            # find a matching decoder state in checkpoint with same rep_dim
            match_idx = None
            if rep_dim is not None:
                for j, dstate in enumerate(dec_states):
                    if j in used_idxs:
                        continue
                    for vk, vv in dstate.items():
                        if isinstance(vv, torch.Tensor) and vv.dim() == 2 and vv.shape[0] == 1 and vv.shape[1] == rep_dim:
                            match_idx = j
                            break
                    if match_idx is not None:
                        break
            # fallback: use next unused state if no match found
            if match_idx is None:
                for j in range(len(dec_states)):
                    if j not in used_idxs:
                        match_idx = j
                        break

            if match_idx is None:
                print(f'No checkpoint decoder state available for decoder[{idx}]')
                continue

            dstate = dec_states[match_idx]
            used_idxs.add(match_idx)
            loaded = 0
            skipped = 0
            for k, v in dstate.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    loaded += 1
                else:
                    skipped += 1
            decoder.load_state_dict(model_sd)
            print(f'Loaded {loaded} params for decoder[{idx}] from checkpoint index {match_idx}, skipped {skipped} mismatched params')

    print('Finished loading weights from {}'.format(filename))


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.scaled_lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate
