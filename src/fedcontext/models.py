import torch
from torch import nn


class MovementGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 6))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class TimingGRUAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(7, 24, batch_first=True)
        self.decoder = nn.GRU(24, 24, batch_first=True)
        self.output = nn.Linear(24, 7)

    def forward(self, x):
        _, z = self.encoder(x)
        repeated = z[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        decoded, _ = self.decoder(repeated, z)
        return self.output(decoded)


class ContextMLPAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 10))

    def forward(self, x):
        return self.net(x)


class ThreeStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.movement = MovementGRU()
        self.timing = TimingGRUAutoencoder()
        self.context = ContextMLPAutoencoder()

    def errors(self, movement, target, timing, context):
        # Three MSEs; no shared parameters between streams.
        m = (self.movement(movement)-target).square().mean(dim=1)
        t = (self.timing(timing)-timing).square().mean(dim=(1, 2))
        c = (self.context(context)-context).square().mean(dim=1)
        return torch.stack([m, t, c], dim=1)
