#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary
from thop import profile
from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes=2, backbone="mobilenet", downsample_factor=16, pretrained=False).to(device)
    summary(model, (3,512,512))
    model = DeepLab(num_classes=2, backbone="xception", downsample_factor=16, pretrained=False)
    input = torch.randn(1, 3, 300, 300)
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)
