import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features,
                                                out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def init_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return x


class GCN(nn.Module):
    def __init__(self, adj_matrix, hidden_features: int = 80,
                 num_embeddings: int = 9, in_features: int = 40,
                 out_features: int = 160):
        super(GCN, self).__init__()

        # Compute the degree matrix and do the normalization
        adj_matrix += torch.eye(adj_matrix.size(0)).to(adj_matrix.device)
        degree_matrix = torch.sum(adj_matrix != 0.0, axis=1)
        inverse_degree_sqrt = torch.diag(torch.pow(degree_matrix, -0.5))

        r"""
        \begin
        D^{\frac{-1}{2}} A D^{\frac{-1}{2}}
        \end
        """
        self.special_matrix = (adj_matrix @ inverse_degree_sqrt).transpose(0, 1) @ inverse_degree_sqrt
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=in_features)

        self.graph_weight_one = GraphConvolution(in_features=in_features,
                                                 out_features=hidden_features,
                                                 bias=False)

        self.graph_weight_two = GraphConvolution(in_features=hidden_features,
                                                 out_features=out_features,
                                                 bias=False)

    def forward(self, x):
        # Before: Shape of x: (batch_size, 9)
        # After: Shape of x: (batch_size, 9, in_features)
        x = self.embedding(x)

        # Go through two-layers GCN
        # Shape of x: (batch_size, 9, hidden_features)
        x = self.special_matrix @ self.graph_weight_one(x)
        x = F.leaky_relu(x, 0.2)

        # Shape of x: (batch_size, 9, output_features)
        x = self.special_matrix @ self.graph_weight_two(x)
        x = F.leaky_relu(x, 0.2)

        return x


if __name__ == "__main__":
    adj_matrix = torch.FloatTensor([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]
    ])

    model = GCN(adj_matrix, num_embeddings=4, hidden_features=80)
    test_tensor = torch.randint(low=0, high=3, size=(4,))
    print(model(test_tensor).shape)
