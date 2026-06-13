import os
import torch
import torch.nn as nn


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class LevelSetNet(nn.Module):
    def __init__(
        self,
        in_dim=2,
        out_dim=1,
        hidden_dim=64,
        num_hidden_layers=3,
        activation='sine',
        pretrained_path=None,
        device='cpu',
        strict=True,
    ):
        super().__init__()
        if pretrained_path is None:
            raise ValueError('LevelSetNet requires a pretrained model.')

        if activation == 'sine':
            act = Sine()
        elif activation == 'tanh':
            act = nn.Tanh()
        else:
            raise ValueError("activation must be 'sine' or 'tanh'")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        layers = [nn.Linear(in_dim, hidden_dim), act]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f'Pretrained model file not found: {pretrained_path}')
        self.load_pretrained_weights(pretrained_path=pretrained_path, device=device, strict=strict)

    def forward(self, x):
        return self.net(x)

    def load_pretrained_weights(self, pretrained_path, device='cpu', strict=True):
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected either a pure state_dict or a dict containing 'model_state_dict'."
            )
        self.load_state_dict(state_dict, strict=strict)
        return self
