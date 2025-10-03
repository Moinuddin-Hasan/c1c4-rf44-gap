# ======================================================================
# FILE: models/model_blocks.py
# PURPOSE: Defines reusable building blocks for neural networks.
# This makes the main model architecture in cnn_model.py cleaner.
# ======================================================================
import torch.nn as nn

def conv_block(in_ch, out_ch, dilation=1):
    """A standard convolutional block: Conv -> ReLU -> BN."""
    pad = dilation  # This padding maintains the feature map size
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(out_ch),
    )