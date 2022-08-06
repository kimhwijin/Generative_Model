from torch import nn
from torch.nn import functional as F

class VanillaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        last_hidden_dim = hidden_dims[-1] * 4
        # Encoder : q(z|x)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim,
                    kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(num_features=h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(last_hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(last_hidden_dim, latent_dim)

        # Decoder : p(x | z)
        modules = []
        in_channels = last_hidden_dim

        self.decoder_input = nn.Linear(latent_dim, in_channels)
        hidden_dims.reverse()
        for h_dim in hidden_dims[:-1]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(num_features=h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.decoder = nn.Sequential(*modules)


