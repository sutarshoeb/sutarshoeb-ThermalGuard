"""
Inception module for YOLOv8 thermal animal detection.
Registered with Ultralytics so YAML config can reference it.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
import ultralytics.nn.modules as ult_modules

class InceptionModule(nn.Module):
    """
    Parallel multi-scale convolution module.

    Four branches run simultaneously on the same input:
      Branch 1 - 1x1 conv: fine detail + channel reduction
      Branch 2 - 3x3 conv: local texture (fur, edges)
      Branch 3 - 5x5 conv: medium spatial (body shape)
      Branch 4 - MaxPool:  dominant thermal intensity

    All four concatenated = rich multi-scale feature map.
    This is the academic novelty — single layer captures
    animal features at all scales simultaneously.
    """

    def __init__(self, c1, c2=None):
        """
        Args:
            c1: input channels (passed automatically by Ultralytics)
            c2: output channels (from YAML args)
        """
        super().__init__()

        # If c2 not specified, keep same channel count as input
        if c2 is None:
            c2 = c1

        # Divide output channels across 4 branches
        b1 = c2 // 4
        b2 = c2 // 4
        b3 = c2 // 8
        b4 = c2 - b1 - b2 - b3

        # Branch 1: 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(c1, b1, kernel_size=1, bias=False),
            nn.BatchNorm2d(b1),
            nn.SiLU(inplace=True)
        )

        # Branch 2: 1x1 reduction -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(c1, b2 // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(b2 // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(b2 // 2, b2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b2),
            nn.SiLU(inplace=True)
        )

        # Branch 3: 1x1 reduction -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(c1, max(b3 // 4, 1), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(b3 // 4, 1)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(b3 // 4, 1), b3, kernel_size=5,
                      padding=2, bias=False),
            nn.BatchNorm2d(b3),
            nn.SiLU(inplace=True)
        )

        # Branch 4: MaxPool -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c1, b4, kernel_size=1, bias=False),
            nn.BatchNorm2d(b4),
            nn.SiLU(inplace=True)
        )

        self.out_channels = b1 + b2 + b3 + b4

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


# -- REGISTER WITH ULTRALYTICS PARSER --------------------------------
import ultralytics.nn.tasks as _tasks
import ultralytics.nn.modules as _modules

# Inject into the tasks module global scope where parse_model looks
_tasks.InceptionModule = InceptionModule
_modules.InceptionModule = InceptionModule

# Patch parse_model's globals directly
import ultralytics.nn.tasks
ultralytics.nn.tasks.__dict__['InceptionModule'] = InceptionModule
