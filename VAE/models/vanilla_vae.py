import torch
from torch import nn
import torch.nn.functional as F

class VanillaVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=100, hidden_dims=[32, 64, 128, 256, 512], image_size=(64, 64)):
        super().__init__()
        module = []
        self.in_channels = in_channels

        for h_dim in hidden_dims:
            module.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, 3, 2, 1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*module)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        self.decoder_stem = nn.Linear(latent_dim, hidden_dims[-1]*4)

        module = []
        in_channels = hidden_dims[-1]
        for h_dim in reversed(hidden_dims[:-1]):
            module.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, h_dim, 3, 2, 1, 1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim
        
        self.decoder = nn.Sequential(*module)
        self.last = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0], 3, 2, 1, 1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[0], self.in_channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, inputs):
        """
        return : [recons, mu, log_var]
        """
        x = self.encoder(inputs)
        valid_shape = x.size()
        
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        x = self.decoder_stem(z)
        
        x = x.view(*valid_shape)
        x = self.decoder(x)
        recons = self.last(x)
        return [recons, mu, log_var]


    @classmethod
    def loss_function(self, recons, inputs, mu, log_var):
        recons_loss = F.mse_loss(recons, inputs)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_weight = 1. / (inputs.size()[1] * inputs.size()[2] * inputs.size()[3])
        loss = recons_loss + kld_weight * kld_loss
        return loss

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return std * eps + mu


        


        
