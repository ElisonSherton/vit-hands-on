# Define all the constants
from model import linearProjection, attentionBlock, multiHeadAttentionBlock, encoderBlock, vit
import torch

torch.manual_seed(0)
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

nc, w, h = 1, 28, 28
S = 4
B = 128
Nseq = (w // S) * (h // S)
d_model = 64
nh = 8
num_classes = 10

model_config = {
    "input_shape": w,
    "num_channels": 1,
    "patch_size": S,
    "num_encoders": 2,
    "cls_token": False,
    "heads_per_encoder": nh,
    "d_model": d_model,
    "act": torch.nn.GELU,
    "expansion_ratio": 2,
    "classifier_mlp_config": [],
    "num_classes": num_classes,
    "debug": False,
    "attn_drop": 0.0,
    "mlp_drop": 0.3,
}


def test_linearProjection():
    input = torch.randn(B, nc, w, h).to(device)
    model = linearProjection(S, d_model, nc)
    model.to(device)
    out = model(input)
    assert out.shape == torch.Size([B, Nseq, d_model])


def test_attentionBlock():
    input = torch.randn(B, Nseq, d_model).to(device)
    model = attentionBlock(d_model, nh)
    model.to(device)
    out = model(input)
    assert out.shape == torch.Size([B, Nseq, d_model // nh])


def test_multiheadAttentionBlock():
    input = torch.randn(B, Nseq, d_model).to(device)
    model = multiHeadAttentionBlock(nh, d_model)
    model.to(device)
    out = model(input)
    assert out.shape == torch.Size([B, Nseq, d_model])

def test_encoderBlock():
    input = torch.randn(B, Nseq, d_model).to(device)
    model = encoderBlock(nh, d_model, 2, torch.nn.GELU)
    model.to(device)
    out = model(input)
    assert out.shape == torch.Size([B, Nseq, d_model])

def test_vit():
    input = torch.randn(B, nc, w, h).to(device)
    model = vit(**model_config)
    model.to(device)
    out = model(input)
    assert out.shape == torch.Size([B, num_classes])
