import torch
import torch.nn as nn

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.temporal1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.graph_conv = nn.Linear(num_nodes, num_nodes)
        self.temporal2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()

    def forward(self, x, A_hat):
        x = self.temporal1(x)
        x = torch.einsum("nctv,vw->nctw", x, A_hat)
        x = self.graph_conv(x)
        x = self.temporal2(x)
        return self.relu(x)

class STGCNRoute(nn.Module):
    def __init__(self, num_nodes, in_channels=1, hidden_channels=16):
        super().__init__()
        self.block = STGCNBlock(in_channels, hidden_channels, num_nodes)
        self.decoder = nn.Linear(hidden_channels, num_nodes)

    def forward(self, x, A_hat):
        x = self.block(x, A_hat)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        flow_pred = self.decoder(x)
        return torch.softmax(flow_pred, dim=-1)