import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from GaborLayer import GaborConv2d
from torchsummary import summary
from torch.autograd import Variable
from deform_conv_V2 import DeformConv2d
import math

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class Conv2d_cd(nn.Module):#中心差分卷积 (CDC) 的稀疏卷积算子，它将 CDC 分别解耦为两个交叉（即水平/垂直 (HV) 和对角线 (DG)）方向的卷积，用于挖掘相互关系和增强局部细节表示
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape

            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]

            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff

class G2C_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(G2C_conv,self).__init__()
        self.conv = nn.Sequential(
            Conv2d_cd(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class G2Cconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(G2Cconv_block,self).__init__()
        self.conv = nn.Sequential(
            Conv2d_cd(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            Conv2d_cd(ch_out, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class G2C_up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(G2C_up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d_cd(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block11(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block11,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class conv_block22(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block22,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        # kernel_size=1,
        x1 = self.conv1(x)

        return x1

class dilaconv_1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(dilaconv_1,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out,kernel_size=3,stride=1,padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(alpha=1.0,inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class dilaconv_2(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(dilaconv_2,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out,kernel_size=3,stride=1,padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(alpha=1.0,inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class dilaconv_4(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(dilaconv_4,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out,kernel_size=3,stride=1,padding=4, dilation=4, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(alpha=1.0,inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1



class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.Conv_1x1(x)
        out = self.conv(x)
        out = residual + out
        return out

class ResConv2d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.Conv_1x1(x)
        out = self.conv(x)
        out = residual + out
        return out

class ResSpatialAtt(nn.Module):#将残差卷积和空间注意力结合
    def __init__(self, ch_in, ch_out):
        super(ResSpatialAtt, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            
        )
        self.SpatialAtt = SpatialAttention(kernel_size=7)
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.Conv_1x1(x)
        out = self.conv(x)
        out = self.SpatialAtt(out)
        out = residual + out
        return out
        
class ChannelAttention(nn.Module):#ratio表示缩放的比例 
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out * x
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)#      
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)  # 对池化完的数据cat 然后进行卷积
        x3 = self.sigmoid(x2)
        x = x3 * x
        return x

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        k = kernel_size
        #t = int(abs(log(C,2) + 1 )/ 2)) #int(abs(log(C,2) + b )/ gamma))
        #k = t if t % 2 else t+1
        self.conv=nn.Conv1d(1,1,kernel_size=k,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

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

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)

class OutlookAttention(nn.Module):#outlook注意力模块
    def __init__(self,dim,num_heads=1,kernel_size=3,padding=1,stride=1,qkv_bias=False,attn_drop=0.1):
        super().__init__()
        self.dim=dim
        self.num_heads=num_heads
        self.head_dim=dim//num_heads
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.scale=self.head_dim**(-0.5)
        
        self.v_pj=nn.Linear(dim,dim,bias=qkv_bias)
        self.attn=nn.Linear(dim,kernel_size**4*num_heads)
        
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(attn_drop)
        
        self.unflod=nn.Unfold(kernel_size,padding,stride) #手动卷积
        self.pool=nn.AvgPool2d(kernel_size=stride,stride=stride,ceil_mode=True) 

    def forward(self, x) :
        B,H,W,C=x.shape
        
        #映射到新的特征v
        v=self.v_pj(x).permute(0,3,1,2) #B,C,H,W
        h,w=math.ceil(H/self.stride),math.ceil(W/self.stride)
        v=self.unflod(v).reshape(B,self.num_heads,self.head_dim,self.kernel_size*self.kernel_size,h*w).permute(0,1,4,3,2) #B,num_head,H*W,kxk,head_dim
        
        #生成Attention Map
        attn=self.pool(x.permute(0,3,1,2)).permute(0,2,3,1) #B,H,W,C
        attn=self.attn(attn).reshape(B,h*w,self.num_heads,self.kernel_size*self.kernel_size,self.kernel_size*self.kernel_size).permute(0,2,1,3,4) #B，num_head，H*W,kxk,kxk
        attn=self.scale*attn
        attn=attn.softmax(-1)
        attn=self.attn_drop(attn)
        
        #获取weighted特征
        out=(attn @ v).permute(0,1,4,3,2).reshape(B,C*self.kernel_size*self.kernel_size,h*w) #B,dimxkxk,H*W
        out=F.fold(out,output_size=(H,W),kernel_size=self.kernel_size,padding=self.padding,stride=self.stride) #B,C,H,W
        out=self.proj(out.permute(0,2,3,1)) #B,H,W,C
        out=self.proj_drop(out)
        
        return out
        
class Res2NetBottleneck(nn.Module):
    expansion = 1  # 残差块的输出通道数=输入通道数*exp+ansion

    def __init__(self, in_channels, out_channels, downsample=None, stride=1, scales=4, groups=1, se=False, norm_layer=True):
        # scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
        super(Res2NetBottleneck, self).__init__()

        if out_channels % scales != 0:  # 输出通道数为4的倍数
            raise ValueError('Planes must be divisible by scales')
        if norm_layer:  # BN层
            norm_layer = nn.BatchNorm2d

        bottleneck_out_channels = groups * out_channels
        self.scales = scales
        self.stride = stride
        self.downsample = downsample
        # 1*1的卷积层,在第二个layer时缩小图片尺寸
        self.iden = nn.Conv2d(in_channels, bottleneck_out_channels, kernel_size=1, stride=stride)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_out_channels, kernel_size=1, stride=stride)
        self.bn1 = norm_layer(bottleneck_out_channels)
        # 3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv2 = nn.ModuleList([nn.Conv2d(bottleneck_out_channels // scales, bottleneck_out_channels // scales,
                                              kernel_size=3, stride=1, padding=1, groups=groups) for _ in
                                    range(scales - 1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_out_channels // scales) for _ in range(scales - 1)])
        # 1*1的卷积层，经过这个卷积层之后输出的通道数变成
        self.conv3 = nn.Conv2d(bottleneck_out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = self.iden(x)

        # 1*1的卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # scales个(3x3)的残差分层架构
        xs = torch.chunk(out, self.scales, 1)  # 将x分割成scales块
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s]))))#这是什么意思
            else:
                ys.append(self.relu(self.bn2[s - 1](self.conv2[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        # 1*1的卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class Gabor_block(nn.Module):
    def __init__(self, ch_in,ch_out):
        super(Gabor_block, self).__init__()
        #self.channel = channel
        self.gabor_conv2d = GaborConv2d(channel_in = ch_in, channel_out = ch_out, kernel_size=3, stride=1, padding=1,
                                        init_ratio=0.5)

    def forward(self, x):  # 128,1,32,32
        x = self.gabor_conv2d(x) # 128*8,20,30,30

        return x
        
class outLine(nn.Module):
    def __init__(self):
        super(outLine, self).__init__()

    def forward(self, x):
        kernel = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).expand(x.size()[1], x.size()[1], 3, 3)
        weight = nn.Parameter(data=kernel, requires_grad=False).to('cuda')
        x = nn.functional.conv2d(x, weight, padding=1)
        return x

        
class up_pro(nn.Module):
    def __init__(self,ch_in,ch_out,):
        super(up_pro,self).__init__()
        self.up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True)
        )
        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ELU(inplace=True)
        )
    def forward(self,x):
        x1 = self.up_1(x)
        x2 = self.up_2(x)
        x3 = torch.cat((x1,x2),dim=1)
        x4 = self.conv(x3)
        return x4

class BN_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(BN_block,self).__init__()
        self.Conv_1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self,x):
        x1 = self.Conv_1(x)
        x2 = self.max(x1)
        x3 = torch.subtract(x1,x2)
        x4= torch.add(x1,x3)
        return x4

class BD_block2(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(BD_block2,self).__init__()
        self.Conv_1 = nn.Conv2d(ch_in,1,kernel_size=3,stride=1,padding=1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self,x):
        x1 = self.Conv_1(x)
        x2 = self.max(x1)
        x3 = torch.subtract(x1,x2)
        #x4= torch.add(x1,x3)
        return x3
        
class Mc_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Mc_block,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
    )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = torch.add(x1,x2)
        x4 = torch.add(x4,x3)
        return x4
        
class D_block(nn.Module):
    def __init__(self):
        super(D_block,self).__init__()
        self.dila1 = dilaconv_1(ch_in=256,ch_out=256)
        self.dila2 = dilaconv_2(ch_in=256,ch_out=256)
        self.dila3_1 = dilaconv_4(ch_in=512,ch_out=256)
        self.dila3_2 = dilaconv_2(ch_in=512,ch_out=256)
        self.dila3_3 = dilaconv_1(ch_in=512,ch_out=256)
        self.conv0 = dilaconv_1(ch_in=768,ch_out=512)
        
    def forward(self,x):
        x1 = self.dila1(x)
        x2_1 = self.dila2(x1)
        x2_2 = self.dila1(x1)
        x2 = torch.cat((x2_1,x2_2),dim=1)
        x3_1 = self.dila3_1(x2)
        x3_2 = self.dila3_2(x2)
        x3_3 = self.dila3_3(x2)
        x3 = torch.cat((x3_1,x3_2,x3_3),dim=1)
        x = self.conv0(x3)
        return x
        
class MLFM1(nn.Module):
    def __init__(self):
        super(MLFM1,self).__init__()
        self.dila0 = dilaconv_1(ch_in=3,ch_out=64)
        self.dila1 = dilaconv_1(ch_in=64,ch_out=64)
        self.dila2 = dilaconv_2(ch_in=64,ch_out=64)
        self.dila3_1 = dilaconv_4(ch_in=128,ch_out=64)
        self.dila3_2 = dilaconv_2(ch_in=128,ch_out=64)
        self.dila3_3 = dilaconv_1(ch_in=128,ch_out=64)
        self.conv0 = dilaconv_1(ch_in=192,ch_out=64)
        
    def forward(self,x):
        x1 = self.dila0(x)
        x2_1 = self.dila2(x1)
        x2_2 = self.dila1(x1)
        x2 = torch.cat((x2_1,x2_2),dim=1)
        x3_1 = self.dila3_1(x2)
        x3_2 = self.dila3_2(x2)
        x3_3 = self.dila3_3(x2)
        x3 = torch.cat((x3_1,x3_2,x3_3),dim=1)
        x = self.conv0(x3)
        return x

class MLFM2(nn.Module):
    def __init__(self):
        super(MLFM2,self).__init__()
        self.dila0 = dilaconv_1(ch_in=64,ch_out=64)
        self.dila1 = dilaconv_1(ch_in=64,ch_out=64)
        self.dila2 = dilaconv_2(ch_in=64,ch_out=64)
        self.dila3_1 = dilaconv_4(ch_in=128,ch_out=64)
        self.dila3_2 = dilaconv_2(ch_in=128,ch_out=64)
        self.dila3_3 = dilaconv_1(ch_in=128,ch_out=64)
        self.conv0 = dilaconv_1(ch_in=192,ch_out=64)
        
    def forward(self,x):
        x1 = self.dila0(x)
        x2_1 = self.dila2(x1)
        x2_2 = self.dila1(x1)
        x2 = torch.cat((x2_1,x2_2),dim=1)
        x3_1 = self.dila3_1(x2)
        x3_2 = self.dila3_2(x2)
        x3_3 = self.dila3_3(x2)
        x3 = torch.cat((x3_1,x3_2,x3_3),dim=1)
        x = self.conv0(x3)
        return x
        
class D_block_2(nn.Module):
    def __init__(self):
        super(D_block_2,self).__init__()
        self.dila1 = dilaconv_1(ch_in=512,ch_out=256)
        self.dila11 = dilaconv_1(ch_in=256,ch_out=256)
        self.dila2 = dilaconv_2(ch_in=256,ch_out=256)
        self.dila3_1 = dilaconv_4(ch_in=512,ch_out=256)
        self.dila3_2 = dilaconv_2(ch_in=512,ch_out=256)
        self.dila3_3 = dilaconv_1(ch_in=512,ch_out=256)
        self.conv0 = dilaconv_1(ch_in=768,ch_out=512)
        
    def forward(self,x):
        x1 = self.dila1(x)
        x2_1 = self.dila2(x1)
        x2_2 = self.dila11(x1)
        x2 = torch.cat((x2_1,x2_2),dim=1)
        x3_1 = self.dila3_1(x2)
        x3_2 = self.dila3_2(x2)
        x3_3 = self.dila3_3(x2)
        x3 = torch.cat((x3_1,x3_2,x3_3),dim=1)
        x = self.conv0(x3)
        return x

class double_deform_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_deform_conv, self).__init__()
        self.conv = nn.Sequential(
            DeformConv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DeformConv2d(out_ch, out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class My_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(My_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        #self.MLFM1 = MLFM1()
        #self.MLFM2 = MLFM2()

        #self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        #self.Conv1 = G2Cconv_block(ch_in=img_ch,ch_out=64)
        #self.Conv1 = ResConv2d(ch_in=img_ch,ch_out=64)
        #self.Mc = Mc_block(ch_in=64,ch_out=64)
        self.BN = BN_block(ch_in=64,ch_out=64)
        #self.Skip1 = outLine()
        self.Conv1 = ResSpatialAtt(ch_in=img_ch,ch_out=64)
        #self.Conv1 = Res2NetBottleneck(in_channels=img_ch, out_channels=64)
        #self.Skip1 = Res2NetBottleneck(in_channels=64, out_channels=64)
        #self.SpatialAtt01 = SpatialAttention(kernel_size=7)
        
        #self.Conv2 = conv_block(ch_in=64,ch_out=128)
        #self.Conv2 = G2Cconv_block(ch_in=64,ch_out=128)
        #self.Conv2 = ResConv2d(ch_in=64,ch_out=128)
        self.Conv2 = ResSpatialAtt(ch_in=64,ch_out=128)
        #self.Conv2 = double_deform_conv(in_ch=64,out_ch=128)
        #self.Conv2 = ResSpatialAtt(ch_in=64,ch_out=128)
        #self.Conv2 = Res2NetBottleneck(in_channels=64, out_channels=128)
        #self.Skip2 = outLine()
        #self.SpatialAtt02 = SpatialAttention(kernel_size=7)  
        
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        #self.Conv3 = ResSpatialAtt(ch_in=128,ch_out=256)
        #self.Conv3 = Res2NetBottleneck(in_channels=128, out_channels=256)
        #self.Conv3 = ResConv2d(ch_in=128,ch_out=256)
        #self.Conv3 = double_deform_conv(in_ch=128,out_ch=256)
        #self.Skip3 = Res2NetBottleneck(in_channels=256, out_channels=256)
        #self.SpatialAtt03 = SpatialAttention(kernel_size=7)
        
        self.conv4_1 = D_block()
        #self.Conv4 = D_block()
        self.conv4_2 = D_block_2()
        #self.Conv4 = conv_block(ch_in=256,ch_out=512)
        #self.Conv4 = ResConv2d(ch_in=256,ch_out=512)
        #self.Conv4 = Res2NetBottleneck(in_channels=256,out_channels=512)
        #self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024,ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        
        #self.eca = ECAAttention(kernel_size=3)
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        #self.Up4 = up_pro(ch_in=512,ch_out=256)
        #self.chatt4 = ChannelAttention(in_planes = 256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        #self.chatt3 = ChannelAttention(in_planes = 128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        #self.chatt2 = ChannelAttention(in_planes = 64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        #self.Up_conv2 = conv_block(ch_in=64, ch_out=64)

        #self.Mc = Mc_block(ch_in=64,ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        # encoding path
        #mlfm = self.MLFM1(x)
        #x1 =self.MLFM1(x)
        #x1 =self.MLFM2(x1)
        x1 = self.Conv1(x)
        #mc = self.Mc(x1)
        #x1_1 = self.Skip1(x1)
        x1_1 = self.BN(x1)
        #x1_1 = self.BN(mlfm)
        #x1_1 = self.BN(mc)
        #x1_1 = self.Mc(x1_1+x1)
        #x1_1=self.SpatialAtt01(x1)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        #x2_1 = self.Skip2(x2)
        #x2_1=self.SpatialAtt02(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        #x3_1 = self.Skip3(x3)
        #x3_1=self.SpatialAtt03(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.conv4_1(x4)
        #x4 = self.Conv4(x4)
        x4 = self.conv4_2(x4)
        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4,d5),dim=1)
        #
        # d5 = self.Up_conv5(d5)
        #
        # d4 = self.Up4(d5)
        d4 = self.Up4(x4)
        #d4 = torch.cat((x3_1,d4),dim=1)
        d4 = torch.cat((x3,d4),dim=1)
        #d4 = self.eca(d4)

        d4 = self.Up_conv4(d4)
        #d4 = self.chatt4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        #d3 = self.eca(d3)
        #d3 = torch.cat((x2_1,d3),dim=1)
        d3 = self.Up_conv3(d3)
        #d3 = self.chatt3(d3)

        d2 = self.Up2(d3)
        #d22 = torch.add(x1_1,d2)
        #d2 = self.Mc(d22)
        d2 = torch.cat((x1_1,d2),dim=1)
        #d2 = self.eca(d2)
        #d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        #d2 = self.chatt2(d2)
        
        #d22 = torch.add(x1_1,d2)
        #d2 = self.Mc(d22)

        d1 = self.Conv_1x1(d2)

        return d1

class GaborNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(GaborNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.Maxpool44 = nn.MaxPool2d(kernel_size=4, stride=4)
        #self.Maxpool88 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.SPAtt = SpatialAttention(kernel_size=7)

        #self.Conv11 = G2C_conv(ch_in=img_ch, ch_out=64)
        self.Conv11_1 = Gabor_block(ch_in=img_ch, ch_out=9)
        self.Conv11_2 = conv_block11(ch_in=img_ch, ch_out=64)
        #self.Conv11_2 = G2C_conv(ch_in=img_ch, ch_out=64)
        self.Conv12 = conv_block22(ch_in=73, ch_out=64)
        #self.Res01 = ResidualBlock(ch_in=73, ch_out=64)
        #self.SpatialAtt01 = SpatialAttention(kernel_size=7)
        
        #self.Conv12 = G2C_conv(ch_in=64, ch_out=64)
        #self.Skip1 = ResidualBlock(ch_in=73, ch_out=73)#将第一个卷积传到残差跳跃连接
        #self.Skip1 = ResidualBlock(ch_in=64, ch_out=64)#将第二个卷积传到残差跳跃连接
        self.Skip1 = Res2NetBottleneck(in_channels=64, out_channels=64)#将第二个卷积传到RES2残差跳跃连接
        #self.Skip1 = outLine()
        
        self.SpatialAtt01 = SpatialAttention(kernel_size=7)
        #self.Skip1 = outline(ch_in=64,ch_out=64)
        #self.Att1 = Attention_block(F_g=64, F_l=64, F_int=64)
        #self.Conv21 = conv_block11(ch_in=64, ch_out=128)
        self.Conv21_1 = Gabor_block(ch_in=64, ch_out=9)
        self.Conv21_2 = conv_block11(ch_in=64, ch_out=128)
        #self.Conv21 = G2C_conv(ch_in=64, ch_out=128)
        self.Conv22 = conv_block22(ch_in=137, ch_out=128)
        #self.Conv22 = G2C_conv(ch_in=128, ch_out=128)
        #self.Res02 = ResidualBlock(ch_in=137, ch_out=128)
        
        #self.Skip2 = ResidualBlock(ch_in=137, ch_out=137)
        #self.Skip2 = ResidualBlock(ch_in=128, ch_out=128)
        #self.Skip2 = Res2NetBottleneck(in_channels=128, out_channels=128)
        self.Skip2 = outLine()
        self.SpatialAtt02 = SpatialAttention(kernel_size=7)  
        
        #self.Att2 = Attention_block(F_g=128, F_l=64, F_int=64)
        self.Conv31 = conv_block11(ch_in=128, ch_out=256)
        #self.Conv31_1 = Gabor_block(ch_in=128, ch_out=9)
        #self.Conv31_2 = conv_block11(ch_in=128, ch_out=256)
        #self.Conv31 = G2C_conv(ch_in=128, ch_out=256)
        self.Conv32 = conv_block22(ch_in=256, ch_out=256)
        #self.Conv32 = G2C_conv(ch_in=256, ch_out=256)
        #self.Res031 = ResidualBlock(ch_in=128, ch_out=256)
        #self.Res032 = ResidualBlock(ch_in=256, ch_out=256)
        
        #self.Skip3 = ResidualBlock(ch_in=256, ch_out=256)
        #self.Skip3 = ResidualBlock(ch_in=256, ch_out=256)
        self.Skip3 = Res2NetBottleneck(in_channels=256, out_channels=256)
        #self.Skip3 = outLine()
        #self.SpatialAtt03 = SpatialAttention(kernel_size=7)
        #self.Att3 = Attention_block(F_g=256, F_l=64, F_int=64)
        
        self.Conv41 = conv_block11(ch_in=256, ch_out=512)
        #self.Conv41_1 = Gabor_block(ch_in=256, ch_out=9)
        #self.Conv41_2 = conv_block11(ch_in=256, ch_out=512)
        #self.Conv41 = G2C_conv(ch_in=256, ch_out=512)
        self.Conv42 = conv_block22(ch_in=512, ch_out=512)
        #self.Conv42 = G2C_conv(ch_in=512, ch_out=512)

        # self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024,ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        #self.Up4 = G2C_up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        #self.Up_conv4 = G2Cconv_block(ch_in=512, ch_out=256)
        self.ChannelAtt1 = ChannelAttention(in_planes = 256)
        #self.SpatialAtt1 = SpatialAttention(kernel_size=7)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        #self.Up3 = G2C_up_conv(ch_in=256, ch_out=128)
        #self.Up_conv3 = conv_block(ch_in=265, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        #self.Up_conv3 = G2Cconv_block(ch_in=256, ch_out=128)
        self.ChannelAtt2 = ChannelAttention(in_planes=128)
        #self.SpatialAtt2 = SpatialAttention(kernel_size=7)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        #self.Up2 = G2C_up_conv(ch_in=128, ch_out=64)
        #self.Up_conv2 = conv_block(ch_in=137, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        #self.Up_conv2 = G2Cconv_block(ch_in=128, ch_out=64)
        self.ChannelAtt3 = ChannelAttention(in_planes=64)
        #self.SpatialAtt3 = SpatialAttention(kernel_size=7)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x11_1 = self.Conv11_1(x)
        x11_2 = self.Conv11_2(x)
        x11 = torch.cat((x11_1, x11_2), dim=1)
        #x11 = x11_1 + x11_2
        #x11 = self.Conv11(x)
        x12 = self.Conv12(x11)
        #x12 = self.Res01(x11)
        #x12 = self.SPAtt(x12)
        #x12 = self.ChannelAtt3(x12)
        #x112 = self.Maxpool(x11)
        
        x21 = self.Maxpool(x12)
        #x112 = self.Att1(g=x21, x=x112)
        #x21 = torch.cat((x112, x21), dim=1)
        x21_1 = self.Conv21_1(x21)
        x21_2 = self.Conv21_2(x21)
        x21 = torch.cat((x21_1, x21_2), dim=1)
        #x21 = self.Conv21(x21)
        x22 = self.Conv22(x21)
        #x22 = self.Res02(x21)
        #x22 = self.ChannelAtt2(x22)
        #x22 = self.SPAtt(x22)

        #x114 = self.Maxpool44(x11)
        x31 = self.Maxpool(x22)
        #x114 = self.Att2(g=x31, x=x114)
        #x31 = torch.cat((x114, x31), dim=1)
        #x31_1 = self.Conv31_1(x31)
        #x31_2 = self.Conv31_2(x31)
        #x31 = torch.cat((x31_1, x31_2), dim=1)
        x31 = self.Conv31(x31)
        x32 = self.Conv32(x31)
        #x31 = self.Res031(x31)
        #x32 = self.Res032(x31)
        #x32 = self.SPAtt(x32)
        #x32 = self.ChannelAtt1(x32)
        
        #x118 = self.Maxpool88(x11)
        x41 = self.Maxpool(x32)
        #x118 = self.Att3(g=x41, x=x118)
        #x41 = torch.cat((x118, x41), dim=1)
        #x41_1 = self.Conv41_1(x41)
        #x41_2 = self.Conv41_2(x41)
        #x41 = torch.cat((x41_1, x41_2), dim=1)
        x41 = self.Conv41(x41)
        x42 = self.Conv42(x41)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4,d5),dim=1)
        #
        # d5 = self.Up_conv5(d5)
        
        # d4 = self.Up4(d5)
        d4 = self.Up4(x42)
        #d4_1 = self.Skip3(x31)
        d4_1 = self.Skip3(x32)
        #d4_11=self.SpatialAtt03(d4_1)
        #d4_11=self.SpatialAtt03(x32)
        
        #d4_2 = self.SPAtt3(d4_1)
        #d4 = torch.cat((d4_11, d4), dim=1)
        d4 = torch.cat((d4_1, d4), dim=1)
        #d4 = torch.cat((x32, d4), dim=1)
        d4 = self.Up_conv4(d4)
        #d4 = self.ChannelAtt1(d4)
        #d4 = self.SpatialAtt1(d4)
        #d4 = self.SPAtt(d4)
        
        d3 = self.Up3(d4)
        #d3_1 = self.Skip2(x21)
        d3_1 = self.Skip2(x22)
        #d3_11=self.SpatialAtt02(d3_1)
        #d3_11=self.SpatialAtt02(x22)
        #d3_2 = self.SPAtt2(d3_1)
        #d3 = torch.cat((d3_11, d3), dim=1)
        d3 = torch.cat((d3_1, d3), dim=1)
        #d3 = torch.cat((x22, d3), dim=1)
        d3 = self.Up_conv3(d3)
        #d3 = self.ChannelAtt2(d3)
        #d3 = self.SpatialAtt1(d3)
        #d3 = self.SPAtt(d3)

        d2 = self.Up2(d3)
        #d2_1 = self.Skip1(x11)
        d2_1 = self.Skip1(x12)
        #d2_11=self.SpatialAtt01(d2_1)
        #d2_2 = self.SPAtt2(d2_1)
        #d2 = torch.cat((d2_11, d2), dim=1) 
        d2 = torch.cat((d2_1, d2), dim=1)
        #d2 = torch.cat((x12, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #d2 = self.ChannelAtt3(d2)
        #d2 = self.SpatialAtt1(d2)
        #d2 = self.SPAtt(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        # self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024,ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4,d5),dim=1)
        #
        # d5 = self.Up_conv5(d5)
        #
        # d4 = self.Up4(d5)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3,d4),dim=1)

        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
