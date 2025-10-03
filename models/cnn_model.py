# models/cnn_model.py
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Corrected lightweight CNN architecture with < 200k parameters.
    """
    def __init__(self):
        super(Net, self).__init__()

        # C1: Input Block -> Output: 32x32x32
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(32)
        )

        # C2: Depthwise Separable Conv -> Output: 32x32x64
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False), # Depthwise
            nn.Conv2d(32, 64, kernel_size=1, bias=False), # Pointwise
            nn.ReLU(), nn.BatchNorm2d(64)
        )

        # C3: Dilated Conv Block -> Output: 32x32x64
        self.c3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2, bias=False), # Dilated
            nn.ReLU(), nn.BatchNorm2d(64)
        )

        # Transition Block using 1x1 Conv -> Output: 32x32x32
        self.t1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(32)
        )

        # C4: Final Conv Block -> Output: 32x32x64
        self.c4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(64)
        )

        # O: Output Block (Classifier Head)
        self.output_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 10, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.t1(x)
        x = self.c4(x)
        x = self.output_block(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)