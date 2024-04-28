import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchinfo import summary


class linearProjection(nn.Module):

    def __init__(self, patch_size: int, d_model: int, num_channels: int):
        """Initial Projection of the image into tokens by breaking into patches
        Args:
            patch_size (int): Size of patch to use break the image into grids
            d_model (int): Embedding dimension of the entire model
            num_channels (int): Number of channels
        """
        super().__init__()
        self.projector = nn.Conv2d(
            in_channels=num_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> B x d_model x H/patch_size x W/patch_size where B = Batch size
        out = self.projector(x)
        N, d_model, *_ = out.shape

        # Use Conv trick to do the linear projection
        out = out.reshape(N, d_model, -1)
        out = out.transpose(1, 2)
        return out


class attentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, attn_drop: float = 0.3):
        """Single Attention Block

        Args:
            d_model (int): Embedding dimension of the model
            num_heads (int): How many heads will be used in the module where this block is used
            attn_drop (float, optional): Dropout to apply to attentions. Defaults to 0.3.
        """
        super().__init__()
        self.Wq = nn.Linear(in_features=d_model, out_features=d_model // num_heads)
        self.Wk = nn.Linear(in_features=d_model, out_features=d_model // num_heads)
        self.Wv = nn.Linear(in_features=d_model, out_features=d_model // num_heads)
        self.d_model = d_model
        self.drop = nn.Dropout(p=attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Here N = num_tokens
        # Input -> N x d_model

        # Parameterizing the individual tokens into Q, K, V
        query = self.Wq(x)  # N x dmodel
        key = self.Wk(x)  # N x dmodel
        value = self.Wv(x)  # N x dmodel

        # N x N shape of attention matrix that will be output
        attentions = F.softmax(
            torch.matmul(query, key.transpose(1, 2)) / self.d_model, dim=-1
        )

        return self.drop(torch.matmul(attentions, value))


class multiHeadAttentionBlock(nn.Module):

    def __init__(self, num_heads: int, d_model: int, attn_drop: float = 0.3):
        """Multi Head Attention Block

        Args:
            num_heads (int): How many heads will be there in each block
            d_model (int): Embedding dimensionality of the model
        """
        super().__init__()
        self.attention_blocks = nn.ModuleList(
            [attentionBlock(d_model, num_heads, attn_drop) for _ in range(num_heads)]
        )
        self.intrahead_mlp = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply all attention heads on input
        attention_outs = [block(x) for block in self.attention_blocks]

        # Concat along the d_model dimension - h vectors each of size d_model/h combine to give d_model
        concatenated_out = torch.cat(attention_outs, dim=-1)
        return self.intrahead_mlp(concatenated_out)


class encoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        expansion_ratio: int,
        act: nn.modules.activation,
        attn_drop: float = 0.3,
        mlp_drop: float = 0.3,
    ):
        """Single Encoder Unit of the transformer

        Args:
            num_heads (int): How many heads in the multihead attention
            d_model (int): Embedding dimensionality of the model
            expansion_ratio (int): Projector MLP at the end of each block
            act (nn.modules.activation): Which non linearity to apply
            attn_drop (float, optional): Dropout to apply on attention blocks. Defaults to 0.3.
            mlp_drop (float, optional): Dropout to apply on MLP. Defaults to 0.3.
        """
        super().__init__()
        self.start_norm = nn.LayerNorm(d_model)
        self.mha = multiHeadAttentionBlock(num_heads, d_model, attn_drop)
        self.middle_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, expansion_ratio * d_model),
            act(),
            nn.Dropout(mlp_drop),
            nn.Linear(expansion_ratio * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x - N * d_model

        # First step of finding attentions
        y = x + self.mha(self.start_norm(x))

        # Applying non linearity
        return y + self.mlp(self.middle_norm(y))


class vit(nn.Module):
    def __init__(
        self,
        input_shape: int,
        num_channels: int,
        patch_size: int,
        num_encoders: int,
        cls_token: bool,
        heads_per_encoder: int,
        d_model: int,
        act: nn.modules.activation,
        expansion_ratio: int,
        classifier_mlp_config: List[int],
        num_classes: int,
        debug: bool,
        attn_drop: float = 0.3,
        mlp_drop: float = 0.3,
    ):
        """_summary_

        Args:
            input_shape (int): Input shape (Assuming square)
            num_channels (int): Number of input channels
            patch_size (int): How big is a single patch (16 x 16 in original ViT paper)
            num_encoders (int): How many encoder blocks to use
            cls_token (bool): Whether to use the CLS token or not
            heads_per_encoder (int): How many heads are used in a single encoder block for MHA
            d_model (int): What is the dimensionality of the model
            act (nn.modules.activation): Non-Linearity to be applied at end of encoder block MLP
            expansion_ratio (int): To decide the model dimensionality for MLP at end of each encoder block
            classifier_mlp_config (List[int]): Classifier MLP configuration
            num_classes (int): How many classes in the classification task
            debug (bool): Whether to print the result of forward pass or not (For debugging purposes)
            attn_drop (float, optional): Dropout to apply for attentions. Defaults to 0.3.
            mlp_drop (float, optional): Dropout to apply to MLP in encoder block. Defaults to 0.3.
        """
        super().__init__()
        self.use_token = cls_token
        self.debug = debug

        # # Define CLS token if it is meant to be used
        if cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, d_model))

        # Make sure the input shape is divisible by patch size without any remainder
        assert (
            input_shape % patch_size == 0
        ), "Input shape should be exactly divisible by patch_size"

        # Define the projection layer
        self.projector = linearProjection(patch_size, d_model, num_channels)

        # Define position embedding table
        tokens = (input_shape // patch_size) ** 2
        self.position_embedding = nn.Embedding(
            num_embeddings=tokens + int(cls_token), embedding_dim=d_model
        )

        # Define encoder blocks
        self.blocks = nn.Sequential(
            *[
                encoderBlock(
                    heads_per_encoder,
                    d_model,
                    expansion_ratio,
                    act,
                    attn_drop,
                    mlp_drop,
                )
                for _ in range(num_encoders)
            ]
        )

        # Define the final classification MLP
        classifier_layers = []
        classifier_mlp_config = [d_model] + classifier_mlp_config + [num_classes]
        for in_feats, out_feats in zip(
            classifier_mlp_config[:-1], classifier_mlp_config[1:]
        ):
            classifier_layers.extend(
                [
                    nn.Linear(in_features=in_feats, out_features=out_feats),
                    act(),
                    nn.Dropout(mlp_drop),
                ]
            )
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print(f"Input Shape: {x.shape}")
        # Get the patch embeddings for stem
        patches = self.projector(x)
        if self.debug:
            print(f"Patches Shape: {patches.shape}")

        # Add CLS token to patches if we are supposed to use it
        if self.use_token:
            N, *_ = x.shape
            cls_ = torch.stack([self.cls_token] * N, dim=0)
            patches = torch.cat([cls_, patches], dim=1)
            if self.debug:
                print(f"Patches with CLS Token shape: {patches.shape}")

        # Add position embeddings to the input only at the beginning

        positions = torch.arange(0, patches.shape[1])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        positions = positions.to(device)
        patches = patches + self.position_embedding(positions)
        x = patches
        x = self.blocks(x)

        if self.debug:
            print(f"Output shape after passing through all the MHA blocks: {x.shape}")

        # If CLS token is used, only use that as the input to classifier
        # Else Average pool all the output values and use that as the input to classifier
        if self.use_token:
            result = self.classifier(x[:, 0, :])
        else:
            result = self.classifier(
                F.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze()
            )
        return result


if __name__ == "__main__":

    # Define a transformer configuration
    num_channels = 1
    input_shape = 224
    patch_size = 16
    num_encoders = 12
    cls_token = False
    heads_per_encoder = 12
    d_model = 24
    act = nn.GELU
    expansion_ratio = 4
    classifier_mlp_config = [4 * d_model]
    num_classes = 20
    attn_drop = 0.3
    mlp_drop = 0.3  # Used for both final classifier and encoder block's MLP

    # Use a random shape and try to do the forward pass
    inp = torch.randn(64, 1, input_shape, input_shape)

    model = vit(
        input_shape,
        num_channels,
        patch_size,
        num_encoders,
        cls_token,
        heads_per_encoder,
        d_model,
        act,
        expansion_ratio,
        classifier_mlp_config,
        num_classes,
        True,
        attn_drop,
        mlp_drop,
    )

    # Print the model summary to have a look at all the parameters
    print(
        summary(model, input_size=(1, input_shape, input_shape), batch_dim=0, verbose=0)
    )

    # Get the output shape
    print(f"Output Shape: {model(inp).shape}")
