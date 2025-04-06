import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_capsules, capsule_dim):
        super(PrimaryCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv(x)
        out = out.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        out = out.permute(0, 3, 1, 2).contiguous().view(batch_size, -1, self.capsule_dim)
        return self.squash(out)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)


class DigitCapsules(nn.Module):
    def __init__(self, num_capsules, in_capsules, in_dim, out_dim):
        super(DigitCapsules, self).__init__()
        self.W = nn.Parameter(0.01 * torch.randn(1, in_capsules, num_capsules, out_dim, in_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        u_hat = torch.matmul(self.W, x).squeeze(-1)
        b_ij = torch.zeros(batch_size, x.size(1), self.W.size(2), 1, device=x.device)

        for _ in range(3):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            b_ij += (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)


class CapsuleNetwork(nn.Module):
    def __init__(self, num_classes=4):
        super(CapsuleNetwork, self).__init__()
        self.cnn = models.resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.primary_capsules = PrimaryCapsules(512, 256, kernel_size=3, stride=1, padding=1, num_capsules=32, capsule_dim=8)
        self.digit_capsules = DigitCapsules(num_classes, 32*7*7, 8, 16)

    def forward(self, x):
        x = self.cnn(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x.norm(dim=-1)


class SiameseCapsuleNetwork(nn.Module):
    def __init__(self):
        super(SiameseCapsuleNetwork, self).__init__()
        self.capsule_net = CapsuleNetwork()

    def forward(self, x1, x2):
        emb1 = self.capsule_net(x1)
        emb2 = self.capsule_net(x2)
        return torch.norm(emb1 - emb2, dim=1, keepdim=True)
