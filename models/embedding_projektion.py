import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)

class ProjectionHead_wLoRa(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, dropout=0.1, alpha=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim),
        )
        # LoRA: rank-reduced adapters
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank  # optional scaling
        # freeze original weights
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_out = self.net(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        out = base_out + lora_out
        return F.normalize(out, p=2, dim=1)


class ProjectionHead_wLoRa_embLayer(nn.Module):
    def __init__(self, in_dim, inner_dim, out_dim, rank=4, dropout=0.1, dropout_inner=0.1, alpha=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, inner_dim),
        )
        # LoRA: rank-reduced adapters
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, inner_dim, bias=False)
        self.scaling = alpha / rank  # optional scaling

        self.dropout_inner = nn.Dropout(dropout_inner)
        self.out_layer = nn.Linear(inner_dim, out_dim)
        
        # freeze original weights
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_out = self.net(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        out = base_out + lora_out
        out = F.normalize(out, p=2, dim=1)
        out = self.dropout_inner(out)
        out = self.out_layer(out)
        return F.normalize(out, p=2, dim=1)


class ProjectionHead_wEmbeddingLayer(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim, dropout=0.1, inter_dropout=0.1, freeze_embedding_layer=False):
        super().__init__()
        # Input embedding network
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, inter_dim),
        )
        # Intermediate dropout and output layer
        self.inter_dropout = nn.Dropout(inter_dropout)
        self.out_layer = nn.Linear(inter_dim, out_dim)

        if freeze_embedding_layer:
            for param in self.net.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, p=2, dim=1)
        x = self.inter_dropout(x)
        x = self.out_layer(x)
        return F.normalize(x, p=2, dim=1)


class ProjectionHead_wConv1d(nn.Module):
    def __init__(self, inp_dim, hidden_channels, kernel_size, out_dim, dropout=0.1):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(in_dim, out_dim),
        # )
        self.inp_dropout = nn.Dropout(dropout)   

        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, stride=kernel_size, padding=0)
        self.conv1_bn = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, 1, kernel_size=kernel_size, stride=kernel_size, padding=0)
        # Determine the dimension after conv layers
        dummy = torch.zeros(1, 1, inp_dim)
        with torch.no_grad():
            out = self.conv2(self.conv1_bn(F.relu(self.conv1(dummy))))
        dim_after_conv = out.shape[-1]
        self.fc = nn.Linear(dim_after_conv, out_dim)

    def forward(self, x):
        x = self.inp_dropout(x)
        # x = self.net(x)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = x.squeeze(1)  # Remove channel dimension
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

class ProjectionHead_wConv1d_wInterDropout(nn.Module):
    def __init__(self, inp_dim, hidden_channels, kernel_size, out_dim, inp_dropout=0.1, inter_dropout=0.1):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(in_dim, out_dim),
        # )
        self.inp_dropout = nn.Dropout(inp_dropout)   

        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, stride=kernel_size, padding=0)
        self.conv1_bn = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, 1, kernel_size=kernel_size, stride=kernel_size, padding=0)
        # Determine the dimension after conv layers
        dummy = torch.zeros(1, 1, inp_dim)
        with torch.no_grad():
            out = self.conv2(self.conv1_bn(F.relu(self.conv1(dummy))))
        dim_after_conv = out.shape[-1]
        self.inter_dropout = nn.Dropout(inter_dropout)
        self.fc = nn.Linear(dim_after_conv, out_dim)

    def forward(self, x):
        x = self.inp_dropout(x)
        # x = self.net(x)
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.conv2(x)
        x = x.squeeze(1)  # Remove channel dimension
        x = self.inter_dropout(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class ProjectionHead_2Layer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, inp_dropout=0.2, activation='relu'):
        super().__init__()
        if activation == 'gelu':
            self.activation = nn.GELU()
            print("Using GELU activation in ProjectionHead_2Layer")
        elif activation == 'relu':
            self.activation = nn.ReLU()
            print("Using ReLU activation in ProjectionHead_2Layer")
        else:
            raise ValueError("Unsupported activation function: {}".format(activation))


        self.input_dropout = nn.Dropout(inp_dropout)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.act1 = self.activation

        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)
