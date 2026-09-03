import torch
from torch import nn
import torchvision.models as tvm

BACKBONES = ("resnet50", "densenet121", "vit_b_16", "mae_face")
MAE_FACE_CKPT: str = "/home/voigt/data/models/mae-face/mae_face_pretrain_vit_base.pth"

def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (headless backbone, feature_dim): (B,3,H,W) -> (B,feat)."""
    
    if name == "mae_face":
        import timm
        m = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0, 
            drop_path_rate=0.1,
            global_pool='avg'
        )
        feat = 768
        
        if pretrained:
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            ckpt = torch.load(MAE_FACE_CKPT, map_location='cpu', weights_only=True)
            m.load_state_dict(ckpt['model'], strict=False)
        m.pos_embed.requires_grad = False    
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        feat = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = m.classifier.in_features
        m.classifier = nn.Identity()
    elif name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = m.heads.head.in_features
        m.heads = nn.Identity()                       # requires exactly 224x224 input
    else:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {name!r}")
    return m, feat