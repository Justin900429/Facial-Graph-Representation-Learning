import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder


class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(kwargs["out_channels"]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DWConv(nn.Module):
    def __init__(self, **kwargs):
        super(DWConv, self).__init__()

        self.block = nn.Sequential(
            ConvBlock(in_channels=kwargs["in_channels"],
                      out_channels=kwargs["in_channels"],
                      kernel_size=kwargs["kernel_size"],
                      padding=kwargs["kernel_size"] // 2,
                      groups=kwargs["in_channels"],
                      bias=False),
            ConvBlock(in_channels=kwargs["in_channels"],
                      out_channels=kwargs["out_channels"],
                      kernel_size=1,
                      bias=False)
        )

    def forward(self, x):
        return self.block(x)


class GraphLearningModel(nn.Module):
    def __init__(self,
                 input_dim: int = 49,
                 forward_dim: int = 128,
                 num_heads: int = 8,
                 head_dim: int = 16,
                 num_layers: int = 6,
                 attn_drop_rate: float = 0.1,
                 proj_drop_rate: float = 0.5,
                 in_channels: int = 30,
                 stride: int = 1,
                 kernel_size: int = 3):
        super(GraphLearningModel, self).__init__()

        # Depth wise convolution for the input
        self.DWConv = nn.Sequential(
            ConvBlock(in_channels=in_channels,
                      out_channels=in_channels,
                      stride=stride,
                      kernel_size=kernel_size,
                      padding=kernel_size // 2,
                      groups=in_channels),
            nn.Flatten(start_dim=2)
        )

        self.eyebrow_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(num_layers)
        ])
        self.mouth_encoder = nn.Sequential(*[
            TransformerEncoder(input_dim=input_dim,
                               forward_dim=forward_dim,
                               num_heads=num_heads,
                               head_dim=head_dim,
                               drop_rate=attn_drop_rate)
            for _ in range(num_layers)
        ])

        self.eyebrow_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(490, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_drop_rate),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

        self.mouth_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(980, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(p=proj_drop_rate),
            nn.Linear(320, 160),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Before: Shape of x: (batch_size, 30, 7, 7)
        # After: Shape of x: (batch_size, 30, 49)
        x = self.DWConv(x)

        # Extract the specific part of vectors
        eyebrow_vector = x[:, :10]
        mouth_vector = x[:, 10:]

        # Shape of eyebrow_vector: (batch_size, 490)
        # Shape of mouth_vector: (batch_size, 980)
        eyebrow_vector = self.eyebrow_encoder(eyebrow_vector)
        mouth_vector = self.mouth_encoder(mouth_vector)

        # Shape of eyebrow_vector: (batch_size, 160)
        # Shape of mouth_vector: (batch_size, 160)
        eyebrow_vector = self.eyebrow_layer(eyebrow_vector)
        mouth_vector = self.mouth_layer(mouth_vector)

        return eyebrow_vector, mouth_vector


if __name__ == "__main__":
    test_vector = torch.randn(1, 30, 7, 7)
    model = GraphLearningModel()

    eyebrow, mouth = model(test_vector)

    print(eyebrow.shape)
    print(mouth.shape)
