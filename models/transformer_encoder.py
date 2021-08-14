import torch
import einops
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, head_dim: int,
                 num_heads: int, drop_rate: float = 0.1):
        super().__init__()

        self.num_heads = num_heads
        self.scale = head_dim ** (-0.5)

        self.q_w = nn.Linear(in_features=input_dim,
                             out_features=head_dim * num_heads,
                             bias=False)
        self.k_w = nn.Linear(in_features=input_dim,
                             out_features=head_dim * num_heads,
                             bias=False)
        self.v_w = nn.Linear(in_features=input_dim,
                             out_features=head_dim * num_heads,
                             bias=False)

        self.dropout = nn.Dropout(p=drop_rate)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)

        self.proj = nn.Linear(in_features=head_dim * num_heads,
                              out_features=input_dim)

    def forward(self, x: torch.tensor) \
            -> torch.tensor:
        # Shape of residual: (batch_size, seq_length, input_dim)
        residual = x

        # Make the dim head
        # Shape of q: (batch_size, num_heads, q_seq_length, head_dim)
        # Shape of k: (batch_size, num_heads, k_seq_length, head_dim)
        # Shape of v: (batch_size, num_heads, v_seq_length, head_dim)
        # NOTE: k_seq_length == v_seq_length
        q = einops.rearrange(self.q_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)
        k = einops.rearrange(self.k_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)
        v = einops.rearrange(self.v_w(x), "b s (n d) -> b n s d",
                             n=self.num_heads)

        # Compute the attention energy
        # Shape of attn: (batch_size, num_heads, q_seq_length, k_seq_length)
        attn = torch.einsum("bnqd,bnkd->bnqk", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # Compute the final weight on value
        # Shape of x: (batch_size, q_seq_length, head_dim * num_heads)
        x = torch.einsum("bnqk,bnkd->bnqd", attn, v)
        x = einops.rearrange(x, "b n q d -> b q (n d)")

        # Shape of x: (batch_size, q_seq_length, input_dim)
        x = self.dropout(self.proj(x)) + residual
        x = self.layer_norm(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 drop_rate: float = 0.1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim,
                      out_features=input_dim),
            nn.Dropout(p=drop_rate)
        )
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        # Shape of residual: (batch_size, input_dim)
        residual = x

        # Before: Shape of x: (batch_size, input_dim)
        # After: Shape of x: (batch_size, input_dim)
        x = self.layer(x)
        x += residual

        x = self.layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, forward_dim: int,
                 num_heads: int, head_dim: int,
                 drop_rate: float = 0.1):
        super().__init__()

        self.attn = MultiHeadAttention(input_dim=input_dim,
                                       head_dim=head_dim,
                                       num_heads=num_heads,
                                       drop_rate=drop_rate)
        self.feedforward = FeedForward(input_dim=input_dim,
                                       hidden_dim=forward_dim,
                                       drop_rate=drop_rate)

    def forward(self, x):
        attn_output = self.attn(x)
        forward_output = self.feedforward(attn_output)

        return forward_output


if __name__ == "__main__":
    test_tensor = torch.randn(1, 30, 49)
    model = TransformerEncoder(input_dim=49,
                               forward_dim=128,
                               num_heads=8,
                               head_dim=16)
    print(model(test_tensor).shape)