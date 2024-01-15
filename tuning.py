from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


DIM = 20
EPOCHS = 10
device = torch.device("cpu")


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=False)

# TA model in sheet3_solution.ipynb
# altered according to https://github.com/cunningham-lab/cb_and_cc/blob/master/cb/utils.py


class BetaDistVAE(nn.Module):
    def __init__(self, hidden_dims=[300, 50, DIM, 50, 300], data_dim=784):
        super().__init__()
        assert len(hidden_dims) == 5, "Insufficiently number of dimensions!"
        self.data_dim = data_dim
        self.device = device

        # self.beta_reg = nn.Parameter(3*torch.ones(1))
        # define IO
        self.in_layer = nn.Linear(data_dim, hidden_dims[0])
        self.out_layer = nn.Linear(hidden_dims[-1], 2*data_dim)
        # self.out_layer_alpha = nn.Linear(hidden_dims[-1], data_dim)
        # self.out_layer_beta = nn.Linear(hidden_dims[-1], data_dim)
        # hidden layer
        self.enc_h = nn.Linear(hidden_dims[0], hidden_dims[1])
        # define hidden and latent
        self.enc_mu = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.enc_sigma = nn.Linear(hidden_dims[1], hidden_dims[2])
        # hidden layer decoder
        self.dec_h = nn.Linear(hidden_dims[2], hidden_dims[-2])
        self.dec_layer = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        self.to(device)

    def encode(self, x: torch.Tensor):
        h1 = F.relu(self.in_layer(x))
        h2 = F.relu(self.enc_h(h1))
        return self.enc_mu(h2), self.enc_sigma(h2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = F.relu(self.dec_h(z))
        h4 = F.relu(self.dec_layer(h3))
        # ALTERED
        beta_params = self.out_layer(h4)
        plus = nn.Softplus()
        alphas = plus(beta_params[:, :self.data_dim])
        betas = 1e-6 + plus(beta_params[:, self.data_dim:])
        # alpha_layer = self.out_layer_alpha(h4)
        # beta_layer = self.out_layer_beta(h4)
        # plus = nn.Softplus()
        # alphas = 1e-6 + plus(alpha_layer)
        # betas = 1e-6 + plus(beta_layer)
        return alphas, betas

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x.view(-1, self.data_dim))
        z = self.reparameterize(mu, logvar)
        alphas, betas = self.decode(z)
        return z, alphas, betas, mu, logvar

# https://github.com/Robert-Aduviri/Continuous-Bernoulli-VAE
# the following function is copied from the above github repo
# where I did not alter anything


def sumlogC(x, eps=1e-5):
    '''
    Numerically stable implementation of 
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 2nd degree approximation

    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    '''
    x = torch.clamp(x, eps, 1.-eps)
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    far_values = torch.log(
        (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far))
    close_values = torch.log(torch.tensor((2.))) + \
        torch.log(1. + torch.pow(1. - 2. * close, 2)/3.)
    return far_values.sum() + close_values.sum()


def beta_loss_function(z, alphas, betas, x, mu, logvar, beta_reg=1, logC_reg=0):
    x = x.view(-1, 784)
    recon_x = alphas / (alphas + betas)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    logC = sumlogC(recon_x)
    return BCE + beta_reg*KLD + logC_reg*logC


def train(model, optimizer, epoch, beta_reg=1, logC_reg=0):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        z, alphas, betas, mu, logvar = model(data)
        loss = beta_loss_function(
            z, alphas, betas, data, mu, logvar, beta_reg, logC_reg)
        loss.backward()
        optimizer.step()


def test(model, optimizer, epoch, beta_reg=1, logC_reg=0):
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            z, alphas, betas, mu, logvar = model(data)

            if i == 0:
                n = min(data.size(0), 8)
                recon_batch = alphas / (alphas + betas)
                recon_batch = recon_batch.view(128, 1, 28, 28)
                comparison = torch.cat([data[:n],
                                        recon_batch[:n]])
                plt.figure(figsize=(10, 4))
                for i in range(1, 2*n+1):
                    ax = plt.subplot(2, n, i)
                    plt.imshow(comparison.cpu().detach().numpy()
                               [i-1, 0, :, :], cmap="gray")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.margins(0, 0)
                plt.savefig(
                    'tuningresults/recon_{:.1f}_{:.1f}.png'.format(beta_reg, logC_reg))
                plt.close()


for beta_reg in np.arange(1, 1.5, 0.1):
    for logC_reg in np.arange(-1, 0, 0.1):
        model = BetaDistVAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        torch.manual_seed(1)
        for epoch in range(1, EPOCHS + 1):
            print("Now Running: beta_reg = {:.1f}, logC_reg = {:.1f}, epoch {:}".format(
                beta_reg, logC_reg, epoch))
            train(model, optimizer, epoch, beta_reg, logC_reg)
            if (epoch == EPOCHS):
                test(model, optimizer, epoch, beta_reg, logC_reg)
                with torch.no_grad():
                    sample = torch.randn(64, DIM).to(device)
                    sample = model.decode(sample)
                    sample = sample[0] / (sample[0] + sample[1])
                    save_image(sample.view(64, 1, 28, 28),
                               'tuningresults/sample_{:.1f}_{:.1f}.png'.format(beta_reg, logC_reg))
