import torch
from torch import nn
from torch.nn import functional as F

class VanillaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims) -> None:
        super().__init__()
        
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.last_hidden_dim = hidden_dims[-1] * 4
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
        
        self.fc_mu = nn.Linear(self.last_hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.last_hidden_dim, latent_dim)

        # Decoder : p(x | z)
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.last_hidden_dim)

        hidden_dims.reverse()
        for i in range(len(hidden_dims)) - 1:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(num_features=hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)

        return [mu, log_var]
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final(result)
        return result
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), inputs, mu, log_var]
    
    def loss_fn(self, recons, inputs, mu, log_var):
        recons_loss = F.mse_loss(recons, inputs)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + 0.005 * kld_loss
        return loss, recons_loss.detach(), kld_loss.detach()

    def generate(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples
    
