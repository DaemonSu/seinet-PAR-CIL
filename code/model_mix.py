import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1D卷积→2D卷积，参数适配（kernel_size/padding/dilation均改为2D元组）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)  # BatchNorm1d→BatchNorm2d
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # 下采样模块同步改为2D版本
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 多尺度分支的1D卷积→2D卷积， dilation同步改为2D
        self.branch1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1))
        self.branch2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2))
        self.branch3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=(3, 3), dilation=(3, 3))
        self.fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=(1, 1))  # 1×1卷积融合通道
        self.bn = nn.BatchNorm2d(in_channels)  # BatchNorm1d→BatchNorm2d
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)  # 通道维度拼接（dim=1）
        out = self.fuse(out)
        return self.relu(self.bn(out))

class FFTBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 2D卷积适配，BatchNorm同步改为2D
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入x: [B, C, H, W]（2D特征图）
        B, C, H, W = x.shape
        # 步骤1：展平空间维度（H×W），适配原FFT逻辑
        x_flat = x.view(B, C, H * W)  # [B, C, H*W]
        # 步骤2：频域变换（沿展平后的空间维度）
        fft_mag = torch.fft.rfft(x_flat, dim=2).abs()  # [B, C, (H*W)//2 + 1]
        # 步骤3：插值回原空间维度长度（H*W）
        fft_mag = F.interpolate(fft_mag, size=H * W, mode='linear', align_corners=True)  # [B, C, H*W]
        # 步骤4：重塑回2D特征图形状
        fft_mag_2d = fft_mag.view(B, C, H, W)  # [B, C, H, W]
        # 步骤5：卷积+归一化+激活
        out = self.conv(fft_mag_2d)
        return self.relu(self.bn(out))

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=1024):
        super().__init__()
        # Stem层：1D→2D卷积，参数适配
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2),  # [B, 64, H//2, W//2]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock(128, 256, stride=2)             # [B, 128, H//4, W//4]
        self.layer2 = nn.Sequential(                               # 深度残差块（2D版本）
            ResidualBlock(256, 256, stride=1),                     # [B, 128, H//8, W//8]
            ResidualBlock(256, 256, stride=1)
        )
        self.layer3 = nn.Sequential(
            MultiScaleBlock(256),                                  # 多尺度块（2D版本，恢复注释）
            nn.Conv2d(256, 512, kernel_size=(1, 1)),               # 1×1卷积提升通道
            nn.BatchNorm2d(512),
            nn.ReLU()
        )                                                          # [B, 256, H//8, W//8]
        self.layer4 = nn.Sequential(
            ResidualBlock(512, 512, stride=2),                     # [B, 256, H//16, W//16]
            FFTBlock(512),                                        # 频域增强（2D版本）
            ResidualBlock(512, 512, stride=2)                      # [B, 256, H//32, W//32]
        )
        self.fc1 = None  # 延迟定义
        self.feature_dim = feature_dim

    def _forward_fc1(self, x):
        if self.fc1 is None:
            in_dim = x.shape[1]
            self.fc1 = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.feature_dim)
            ).to(x.device)
        return F.normalize(self.fc1(x), dim=-1)

    def forward_to_bottleneck(self, x, stage="layer4"):
        x = x.permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if stage == "layer3":
            return x
        x = self.layer4(x)
        x = x.flatten(1)
        if stage == "layer4":
            return x
        raise ValueError(f"Unsupported bottleneck stage: {stage}")

    def forward_from_bottleneck(self, x, stage="layer4"):
        if stage == "layer3":
            x = self.layer4(x)
            x = x.flatten(1)
        elif stage != "layer4":
            raise ValueError(f"Unsupported bottleneck stage: {stage}")
        return self._forward_fc1(x)

    def forward(self, x):
        # 输入x原始形状：[B, 29, 42, 3]（B=批量，H=29，W=42，C=3）
        # 转换为2D卷积要求的格式：[B, C, H, W]
        x = x.permute(0, 3, 1, 2)  # [B, 3, 29, 42]
        # x = x.transpose(1, 2)  # [B, 3, 7000]
        # 特征提取流程（所有模块已改为2D版本）
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.flatten(1)  # [B, 256*1*2] = [B, 512]

        # 自动初始化 fc1
        if self.fc1 is None:
            in_dim = x.shape[1]
            self.fc1 = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.feature_dim)
            ).to(x.device)

        feat = F.normalize(self.fc1(x), dim=-1)
        return feat


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=1024, num_classes=10):
        super().__init__()
        # 分类头无需修改（输入为1D特征向量，与原逻辑一致）
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
