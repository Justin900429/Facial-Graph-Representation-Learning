import torch
import torch.nn as nn


class AUFusion(nn.Module):
    def __init__(self, num_classes, in_features=9):
        super(AUFusion, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=num_classes)
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-6)

    def forward(self, eyebrow, mouth, gcn):
        # Shape of gcn: (9, 160)
        # Shape of eyebrow_gcn: (160, 3)
        # Shape of mouth_gcn: (160, 6)
        gcn = gcn.transpose(0, 1)
        eyebrow_gcn, mouth_gcn = gcn.split([3, 6], dim=1)

        # Before: Shape of eyebrow: (batch_size, 160)
        # Before: Shape of mouth: (batch_size, 160)
        # After: Shape of eyebrow: (batch_size, 3)
        # After: Shape of mouth: (batch_size, 6)
        eyebrow = (eyebrow @ eyebrow_gcn)
        mouth = (mouth @ mouth_gcn)

        # Shape of features: (batch_size, 9)
        features = torch.cat([eyebrow, mouth], dim=-1)
        features = self.layer_norm(features)

        # Shape of features: (batch_size, num_classes)
        features = self.linear(features)

        return features


if __name__ == "__main__":
    eyebrow_test = torch.randn(1, 160)
    mouth_test = torch.randn(1, 160)
    gcn_test = torch.randn(9, 160)

    model = AUFusion(num_classes=5)
    print(model(eyebrow_test, mouth_test, gcn_test).shape)
