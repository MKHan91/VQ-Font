import torch
import torch.nn as nn
import torchvision.models as models


def gram_matrix(feature):
    (b, ch, h, w) = feature.size()
    features = feature.view(b, ch, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (ch * h * w)


class VGGFeatureExtractor(nn.Module):
    """VGG16 feature extractor for style loss (returns conv1_1, conv2_1, conv3_1, conv4_1)
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        # slice layers to get outputs at desired conv layers
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(0, 4)])   # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])   # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)])  # relu3_3
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(16, 23)]) # relu4_3
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

        # self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x):
        # x = (x - self.mean) / self.std
        h = x
        h1 = self.slice1(h)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return [h1, h2, h3, h4]


class StyleLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.vgg = VGGFeatureExtractor().to(device)

    def forward(self, input_img, target_img):
        """Compute style loss between input and target images using Gram matrices.

        input_img and target_img expected in range [0,1] and shape [B, C, H, W].
        """
        # VGG expects 3-channel images; if single-channel, repeat
        if input_img.shape[1] == 1:
            input_ = input_img.repeat(1, 3, 1, 1)
            target_ = target_img.repeat(1, 3, 1, 1)
        else:
            input_ = input_img
            target_ = target_img

        feats_in = self.vgg(input_)
        feats_tar = self.vgg(target_)

        loss = 0.0
        for f_in, f_tar in zip(feats_in, feats_tar):
            G_in = gram_matrix(f_in)
            G_tar = gram_matrix(f_tar)
            loss += nn.functional.mse_loss(G_in, G_tar)

        return loss
