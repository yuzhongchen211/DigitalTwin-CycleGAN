import torch
import torch.nn as nn
import timm

class Encoder(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        backbone = timm.create_model(backbone_name, pretrained=True)
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.embed_dim = self.backbone.num_features

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(2 * B, C // 2, H, W)
        x = self.backbone(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if isinstance(drop, tuple):
            drop_probs = drop
        else:
            drop_probs = (drop, drop)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DetModel(nn.Module):
    def __init__(self, out_dim, backbone_name='vit_small_patch16'):
        super().__init__()
        self.backbone = Encoder(backbone_name)
        self.head = Mlp(self.backbone.embed_dim, out_features=out_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.head(x)
        x = x.reshape(B,-1)
        return x
if __name__=='__main__':
    model = DetModel(4,'swin_small_patch4_window7_224')
    state = torch.rand((2,6,224,224))
    out = model(state)
    print(out.shape)