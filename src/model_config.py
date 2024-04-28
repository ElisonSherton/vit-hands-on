import torch.nn as nn

config = {
    "input_shape": 28,
    "num_channels": 1,
    "patch_size": 4,
    "num_encoders": 3,
    "cls_token": False,
    "heads_per_encoder": 8,
    "d_model": 64,
    "act": nn.GELU,
    "expansion_ratio": 2,
    "classifier_mlp_config": [],
    "num_classes": 10,
    "debug": False,
    "attn_drop": 0.0,
    "mlp_drop": 0.3,
}
