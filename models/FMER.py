import torch
import torch.nn as nn
from .graph_model import GraphLearningModel
from .au_gcn import GCN
from .au_fusion import AUFusion


class FMER(nn.Module):
    def __init__(self, adj_matrix, num_classes,
                 device, hidden_features=80):
        super(FMER, self).__init__()
        self.graph = GraphLearningModel()
        self.au_gcn = GCN(adj_matrix=adj_matrix,
                          hidden_features=hidden_features)
        self.au_fusion = AUFusion(num_classes=num_classes)

        # Used to train the embedding
        self.au_seq = torch.arange(9).to(device)

    def forward(self, patches):
        batch_size = patches.size(0)

        # Node learning and edge learning
        # Shape of patches: (batch_size, 30, 7, 7)
        eyebrow, mouth = self.graph(patches)

        # Training the GCN
        # Shape of au_seq: (9)
        # Shape of gcn_output: (9, 160)
        gcn_output = self.au_gcn(self.au_seq)

        # Fuse the graph learning and GCN
        fusion_output = self.au_fusion(eyebrow, mouth, gcn_output)

        return fusion_output


if __name__ == "__main__":
    test_tensor = torch.rand(1, 30, 7, 7)
    adj_matrix = torch.rand(9, 9)
    model = FMER(adj_matrix=adj_matrix,
                 num_classes=5,
                 device="cpu")

    print(model(test_tensor).shape)