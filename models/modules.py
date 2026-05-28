import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    
    return P

class UNetSegmentationHead(nn.Module):
    """Fully fixed U-Net segmentation head - resolves all dimension mismatches"""
    
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        
        # encoder
        self.encoder1 = self._conv_block(in_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        
        # decoder - fix: ensure all channel dimensions match
        self.decoder1 = self._up_conv_block(256, 128)  # in 256, out 128
        self.decoder2 = self._up_conv_block(256, 64)   # in 256 (128+128), out 64
        
        # fix: final_conv input channels should be 128, not 64
        # because d2 concatenation gives 64+64=128 channels
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _up_conv_block(self, in_channels, out_channels):
        """Ensure transposed conv input/output channel dimensions are correct"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # encoder
        e1 = self.encoder1(x)        # [B, 64, H, W]
        e2 = self.encoder2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.encoder3(self.pool(e2))  # [B, 256, H/4, W/4]
        
        # decoder
        d1 = self.decoder1(e3)       # [B, 128, H/2, W/2]
        d1 = torch.cat((e2, d1), dim=1)  # [B, 256, H/2, W/2] (128+128)
        
        d2 = self.decoder2(d1)       # [B, 64, H, W]
        d2 = torch.cat((e1, d2), dim=1)  # [B, 128, H, W] (64+64)
        
        # output - now 128 input channels, matching final_conv
        out = self.final_conv(d2)    # [B, 1, H, W]
        out = self.sigmoid(out)
        
        return out

class EdgeDetector(nn.Module):
    """Edge detection module - fixed version"""
    
    def __init__(self):
        super().__init__()
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # fix: use float data type
        edge_kernel = torch.tensor([
            [-1.0, -1.0, -1.0],
            [-1.0, 8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.edge_conv.weight.data = edge_kernel
        # optional: freeze weights if no update needed
        # self.edge_conv.weight.requires_grad = False
    
    def forward(self, x):
        edges = self.edge_conv(x)
        edges = torch.sigmoid(edges)
        return edges

# test fix
if __name__ == "__main__":
    try:
        detector = EdgeDetector()
        print("EdgeDetector initialized successfully!")
        
        # test forward pass
        test_input = torch.randn(1, 1, 256, 256)
        output = detector(test_input)
        print(f"Output shape: {output.shape}")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Error: {e}")