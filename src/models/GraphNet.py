import torch.nn as nn
from gatmodule import ContourMap, RegionMap, SEGRModule, PSPModule
import timm



class GraphNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('res2net50_26w_4s.in1k',features_only = True, 
                                         out_indices=(0,1,2,3,4), pretrained=True, pretrained_cfg_overlay=dict(file = 'checkpoints/res2net50_26w_4s/pytorch_model.bin'))

        # Exteact edge features from x1,x2
        self.contour = ContourMap(low_channels=64, high_channels=256)
        # Extract region features from x3,x4,x5
        self.region = RegionMap(low_channels=512, mid_channels=1024, high_channels=512)
        
        self.spp = PSPModule(features=2048)

        # graph reasoning
        self.cgr = SEGRModule(256, 32, 8)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), # [1,128,96,96]

            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 1x1
            nn.BatchNorm2d(64),
           
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'), # [1,64,192,192]

            # nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=1),  # 1x1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'), # # [1,32,384,384]

            nn.Conv2d(32, 16, kernel_size=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=1), 
        )

        # [1, 32, 192, 192]
        self.edge_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 2, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(2),
            nn.Softmax(dim=1)
        )

       

    def forward(self, x):
    
        # encoder
        x1, x2, x3, x4, x5 = self.encoder(x)

        
        x5 = self.spp(x5)
        x_c = self.contour(x1, x2)
        x_r = self.region(x3, x4, x5)

        x_g = self.cgr(x_r, x_c)

        edge_pred = self.edge_conv(x_c)
        region_pred = self.decoder(x_g)

        return edge_pred, region_pred
