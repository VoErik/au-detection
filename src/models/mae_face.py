import torch
import timm
from torch import nn

MAE_FACE_CKPT: str = "/home/voigt/data/models/mae-face/mae_face_pretrain_vit_base.pth"

def build_mae_face(
    n_classes: int, 
    pretrained: bool = True, 
    freeze_backbone: bool = False, 
    dropout: float = 0.0,
    ckpt_path: str = MAE_FACE_CKPT
) -> nn.Module:
    """Builds a timm-based MAE-Face ViT and handles pretrained weight loading."""
    
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=n_classes,
        drop_rate=dropout,
        drop_path_rate=0.1,
        global_pool='avg'
    )
    
    if pretrained:
        import argparse
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.pos_embed.requires_grad = False 
        
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('head.'):
                param.requires_grad = False
                
    return model