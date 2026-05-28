import timm
import torch
import torch.nn as nn


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


class ResNetDFRMBackbone(nn.Module):

    def __init__(self, base_arch='resnet50', out_indices=None, pretrained=True, se_reduction=16, disable_dfrm=False):
        super().__init__()
        if out_indices is None:
            out_indices = [1, 2, 3]
        # instantiate timm features_only backbone
        self.base_arch = base_arch
        self.model = timm.create_model(base_arch, features_only=True, out_indices=[i-1 for i in out_indices], pretrained=pretrained)
        # feature_info from timm
        self.feature_info = self.model.feature_info
        self.disable_dfrm = bool(disable_dfrm)
        # create DFRM channel modulator layers for each output channel
        channels = self.feature_info.channels()
        self.dfrm_modules = nn.ModuleList([DFRMChannelModulator(c, reduction=se_reduction) for c in channels])

    def forward(self, x):
        feats = self.model(x)
        if self.disable_dfrm:
            return feats
        out = []
        for i, f in enumerate(feats):
            f = self.dfrm_modules[i](f)
            out.append(f)
        return out

    def forward_with_dfrm_intermediates(self, x):
        
        pre_feats = self.model(x)
        if self.disable_dfrm:
            return pre_feats, pre_feats
        post_feats = [self.dfrm_modules[i](f) for i, f in enumerate(pre_feats)]
        return pre_feats, post_feats

    def get_dfrm_channel_importance(self):
        
        imps = []
        for dfrm in self.dfrm_modules:
            w = dfrm.fc[2].weight.detach().cpu()
            imps.append(w.abs().sum(dim=1))
        return imps


__all__ = ['ResNetDFRMBackbone']
