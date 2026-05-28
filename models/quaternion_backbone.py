import torch
import torch.nn as nn


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = channels

    def channels(self):
        return self._channels


class DFRMChannelModulator(nn.Module):
    

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class QuaternionBackbone(nn.Module):

    def __init__(self, out_indices=None, pretrained=False, in_ch=3, se_reduction=16, disable_dfrm=False):
        super().__init__()
        if out_indices is None:
            out_indices = [1, 2, 3]
        # define simple encoder blocks
        chs = [64, 128, 256, 512]
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(chs[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        def _make_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.layer1 = _make_block(chs[0], chs[0])
        self.layer2 = _make_block(chs[0], chs[1])
        self.layer3 = _make_block(chs[1], chs[2])
        self.layer4 = _make_block(chs[2], chs[3])

        # DFRM channel modulators for each stage
        self.dfrm_modules = nn.ModuleList([DFRMChannelModulator(c, reduction=se_reduction) for c in chs])
        self.disable_dfrm = bool(disable_dfrm)

        # choose channels for requested out indices
        available = [chs[0], chs[1], chs[2], chs[3]]
        selected = []
        for idx in out_indices:
            i = max(1, min(len(available), int(idx))) - 1
            selected.append(available[i])

        self.feature_info = _FeatureInfo(selected)

    def forward(self, x):
        feats = []
        x = self.stem(x)
        x1 = self.layer1(x)
        if not self.disable_dfrm:
            x1 = self.dfrm_modules[0](x1)
        feats.append(x1)
        x2 = self.layer2(x1)
        if not self.disable_dfrm:
            x2 = self.dfrm_modules[1](x2)
        feats.append(x2)
        x3 = self.layer3(x2)
        if not self.disable_dfrm:
            x3 = self.dfrm_modules[2](x3)
        feats.append(x3)
        x4 = self.layer4(x3)
        if not self.disable_dfrm:
            x4 = self.dfrm_modules[3](x4)
        feats.append(x4)

        # return only requested feature maps in order (features_only behavior)
        out = []
        for idx in range(len(self.feature_info.channels())):
            out.append(feats[idx])
        return out

    def forward_with_dfrm_intermediates(self, x):
    
        pre = []
        post = []
        x = self.stem(x)
        x1 = self.layer1(x)
        pre.append(x1)
        p1 = self.dfrm_modules[0](x1) if not self.disable_dfrm else x1
        post.append(p1)
        x2 = self.layer2(p1)
        pre.append(x2)
        p2 = self.dfrm_modules[1](x2) if not self.disable_dfrm else x2
        post.append(p2)
        x3 = self.layer3(p2)
        pre.append(x3)
        p3 = self.dfrm_modules[2](x3) if not self.disable_dfrm else x3
        post.append(p3)
        x4 = self.layer4(p3)
        pre.append(x4)
        p4 = self.dfrm_modules[3](x4) if not self.disable_dfrm else x4
        post.append(p4)
        # return only requested indices
        out_pre = []
        out_post = []
        for idx in range(len(self.feature_info.channels())):
            out_pre.append(pre[idx])
            out_post.append(post[idx])
        return out_pre, out_post

    def get_dfrm_fc2_weights(self):
        """Return DFRM fc2 (expansion) weight tensor for each layer.

        Returns a list of tensors, one per stage, shape (C, C//r).
        """
        return [dfrm.fc[2].weight.detach().cpu().clone() for dfrm in self.dfrm_modules]

    def get_dfrm_channel_importance(self):
        
        imps = []
        for dfrm in self.dfrm_modules:
            w = dfrm.fc[2].weight.detach().cpu()  # (C, C//r)
            imp = w.abs().sum(dim=1)
            imps.append(imp)
        return imps


__all__ = ['QuaternionBackbone']
