import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
abn = nn.BatchNorm2d

class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, abn = abn, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.abn = abn
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            abn(out_features),
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = self.abn(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [ F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle



class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

class M2Skip(nn.Module):
    def __init__(self, in_channels=[16,32],model_type='bottom'):#大,小
        super(M2Skip, self).__init__()
    
        self.convl=nn.Sequential(
                nn.Conv2d(in_channels[0],in_channels[1],3,2,1),
                nn.BatchNorm2d(in_channels[0]),
                #nn.ReLU(inplace=True),
                nn.GELU(),
                #nn.AdaptiveAvgPool2d((32, 32)),
                #nn.BatchNorm2d(in_channels[1]),
            )
        self.convs=nn.Sequential(
            
            nn.Conv2d(in_channels[1], in_channels[1], 3,1,1),
            #nn.BatchNorm2d(in_channels[1]),
            nn.BatchNorm2d(in_channels[1]),
            #nn.ReLU(inplace=True),
            nn.GELU(),
            )
        self.fuse_conv = nn.Sequential(nn.Conv2d(2 * in_channels[1], in_channels[1], 3,1,1),
                                        nn.BatchNorm2d(in_channels[1]),
                                        nn.GELU()
                                        #SPBlock(in_channels[1],in_channels[1])
                                        )

    def forward(self, xl, xs):
        xl=self.convl(xl)
        xs=self.convs(xs)
        x=torch.cat([xl,xs],dim=1)
        x=self.fuse_conv(x)

        return x


class ContourMap(nn.Module):
    def __init__(self, low_channels=64, high_channels=160):
        super(ContourMap, self).__init__()
        
        self.conv1=nn.Sequential(
            nn.Conv2d(low_channels, low_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.GELU(),
            nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(low_channels),
            nn.GELU()
            )
        
        self.conv2=nn.Sequential(
            nn.Conv2d(high_channels, low_channels, kernel_size=1),
            nn.BatchNorm2d(low_channels),
            nn.GELU(),
            nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1, bias=False),  # 3x3
            nn.BatchNorm2d(low_channels),
            nn.GELU()
            )
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(2 * low_channels, low_channels, 3,1,1),
            nn.BatchNorm2d(low_channels),
            nn.GELU(),
            
            nn.Conv2d(low_channels, 32, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(32),
            nn.GELU(),

            # nn.Softmax(dim=1)
            )

        self.edge_conv = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=1, bias=False),  # 1x1
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

    def forward(self, x1, x2):
        
        # x2 上采样
        x2_up = F.interpolate(x2, size=(192, 192), mode='bilinear', align_corners=True)
        x1_cov = self.conv1(x1)
        x2_conv = self.conv2(x2_up)

        xc = torch.concat((x1_cov, x2_conv), dim=1)
        edge = self.fuse_conv(xc)
        # edge_map = self.out_conv(x_125) 
        return edge

# x3,x4,x5
class RegionMap(nn.Module):
    def __init__(self, low_channels, mid_channels, high_channels):
        super(RegionMap, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv2d(low_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(high_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        
        self.att1 = ShuffleAttention(channel=mid_channels, G = 8)
        self.att2 = ShuffleAttention(channel=low_channels, G = 8)

        self.conv45 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(mid_channels, low_channels, kernel_size=1),
            nn.BatchNorm2d(low_channels),
            nn.ReLU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(low_channels, low_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(),
            nn.Conv2d(low_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Conv2d(256, 128, kernel_size=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU()
        )

    def forward(self, x3, x4, x5):
        x5 = self.conv5(x5)
        x45 = x4 + self.att1(x4 + x5)
        x345 = x3 + self.att2(x3 + self.conv45(x45))

        region = self.out_conv(x345)
        return region

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, bias= False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias = False)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h

def adj_index(h, k, node_num):
    dist = torch.cdist(h, h, p=2)
    each_adj_index = torch.topk(dist, k, dim=2).indices
    adj = torch.zeros(
        h.size(0), node_num, node_num, 
        dtype=torch.int, device=h.device, requires_grad=False
    ).scatter_(dim=2, index=each_adj_index, value=1)
    return adj


'''
    Synergistic Edge-guided Graph Reasoning Module
'''
class SEGRModule(nn.Module):
    def __init__(self, 
                 num_state, 
                 num_state_mid, 
                 mids, 
                 abn=nn.BatchNorm2d, 
                 normalize=False, 
                 ):
        super(SEGRModule, self).__init__()

        # cgr
        self.normalize = normalize
        self.num_s = 2*int(num_state_mid) # state个数，与edge对齐
        self.num_n = (mids) * (mids) # 节点个数
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_state, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_state, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # self.reduce_dim = nn.Conv2d(2*num_state, self.num_s, kernel_size=1)
        self.conv_extend = nn.Conv2d(self.num_s, num_state, kernel_size=1, bias=False)

        self.blocker = abn(num_state)

    def forward(self, x, edge):
        # Projection 
        edge = F.upsample(edge, size=(x.size()[-2], x.size()[-1]))
        n, c, h, w = x.size()

        x = x.view(n, 256, 48, -1)
        # =====================

        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # Parallel processing for region and edge features
        x_proj_region = self.conv_proj(x)
        x_proj_edge = self.conv_proj(x) * edge # x_proj_edge = self.conv_proj(edge) * edge

        # Integrate region and edge features with attention mechanism
        x_proj_combined = (x_proj_region + x_proj_edge).view(n, self.num_s, -1)
        x_proj_combined = torch.nn.functional.softmax(x_proj_combined, dim=-1)
        # =========================

        x_n_state = torch.matmul(x_state_reshaped, x_proj_combined.permute(0, 2, 1)) # [2,32,32] batch, num_state, num_node
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_proj_combined)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.blocker(self.conv_extend(x_state))
        return out
    