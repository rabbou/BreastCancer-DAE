from __future__ import print_function
from __future__ import division
import torch
import torch.utils.data as tud
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from utils import *

class L1Penality(torch.autograd.Function):

    """
    In the forward pass we receive a Tensor containing the input and return
    a Tensor containing the output. ctx is a context object that can be used
    to stash information for backward computation. You can cache arbitrary
    objects for use in the backward pass using the ctx.save_for_backward method.
    """
    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input
  
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone().sign().mul(ctx.l1weight)
        grad_input += grad_output
        return grad_input, None

class DAE(nn.Module):
    def __init__(self, n_comp):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, n_comp, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(n_comp, n_comp, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2, padding=0)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_comp, n_comp, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_comp, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU()
        )

    def forward(self, input, sparse_constraint):
        out, ind = self.encoder(input)
        if sparse_constraint:
            out = L1Penality.apply(out, 0.1)
        out = self.unpool(out, ind)
        out = self.decoder(out)
        return out

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch']

class AutoEncoder(object):
    """AutoEncoder (AE)

    Parameters
    ----------
    n_inputs: int, feature size of input data
    n_components: int, feature size of output
    lr: float, learning rate (default: 0.001)
    batch_size: int, batch size (default: 512)
    path: string, path to save trained model (default: "vae.pth")
    """
    def __init__(self, n_inputs, patch_size=64, lr=1.0e-3, batch_size=512, noise_strength=25, cuda=True,
                 n_components=100, path="dae.pt", load_weights=False, sparse_constraint=False, plot=True):
        self.model = nn.DataParallel(DAE(n_components))
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.lr = lr
        self.path = path
        self.noise_strength = noise_strength
        self.trans = transforms.ToTensor()
        self.load_weights = load_weights
        self.sparse_constraint = sparse_constraint
        self.plot = plot
        self.initialize()

    def fit(self, Xr, Xd, epochs):
        """Fit VAE from data Xr
        Parameters
        ----------
        :in:
        Xr: 2d array of shape (n_data, n_dim). Training data
        Xd: 2d array of shape (n_data, n_dim). Dev data, used for early stopping
        epochs: int, number of training epochs
        """
        train_loader = tud.DataLoader(noisy_unlabelled_breakHis(Xr, self.noise_strength, self.trans),
                                      batch_size=self.batch_size, shuffle=True)
        dev_loader = tud.DataLoader(noisy_unlabelled_breakHis(Xd, self.noise_strength, self.trans),
                                    batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=1e-3)
        start_epoch = 1
        if self.load_weights:
            model, optimizer, scheduler, start_epoch = load_ckp(self.path, self.model, optimizer, scheduler)
            
        best_dev_loss = np.inf
        for epoch in range(start_epoch, epochs + start_epoch):
            train_loss = self._train(train_loader, optimizer)
            dev_loss, _ = self._evaluate(dev_loader)
            if dev_loss < best_dev_loss:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': self.model,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, self.path)
                best_dev_loss = dev_loss
            print('Epoch: %d, lr: %.0e, train loss: %.4f, dev loss: %.4f' % (
                epoch, optimizer.param_groups[0]['lr'], train_loss, dev_loss))
            scheduler.step(dev_loss)
        return

    def transform(self, X):
        """Transform X
        Parameters
        ----------
        :in:
        X: 2d array of shape (n_data, n_dim)
        :out:
        Z: 2d array of shape (n_data, n_components)
        """
        try:
            self.model = torch.load(self.path)['model']
        except Exception as err:
            print("Error loading '%s'\n[ERROR]: %s\nUsing initial model!" % (self.path, err))
        test_loader = tud.DataLoader(noisy_unlabelled_breakHis(X, self.noise_strength, self.trans),
                                     batch_size=self.batch_size, shuffle=False)
        _, Z = self._evaluate(test_loader)
        return Z

    def _train(self, train_loader, optimizer):
        self.model.train()
        train_loss = 0
        for batch_idx, (data_clean, data_noisy) in enumerate(train_loader):
            data_noisy = data_noisy.to(self.device, dtype=torch.float)
            data_clean = data_clean.to(self.device, dtype=torch.float)
            optimizer.zero_grad()
            recon_batch = self.model(data_noisy, self.sparse_constraint)
            if batch_idx == 0 and self.plot:
                plot_samples(recon_batch.cpu().detach().numpy())
                plot_samples(data_clean.cpu().detach().numpy())
            loss = self._loss_function(recon_batch, data_clean)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        return train_loss/(batch_idx+1)

    def _evaluate(self, loader):
        self.model.eval()
        loss = 0
        fs = []
        with torch.no_grad():
            for batch_idx, (data_clean, data_noisy) in enumerate(loader):
                data_noisy = data_noisy.to(self.device, dtype=torch.float)
                data_clean = data_clean.to(self.device, dtype=torch.float)
                recon_batch = self.model(data_noisy, self.sparse_constraint)
                loss += self._loss_function(recon_batch, data_clean)
                fs.append(recon_batch)
        fs = torch.cat(fs).cpu().numpy()
        return loss/(batch_idx+1), fs

    def _loss_function(self, recon_x, x):
        """VAE Loss
        Parameters
        ----------
        :in:
        recon_x: 2d tensor of shape (batch_size, n_dim), reconstructed input
        x: 2d tensor of shape (batch_size, n_dim), input data
        mu: 2d tensor of shape (batch_size, n_components), latent mean
        logvar: 2d tensor of shape (batch_size, n_components), latent log-variance
        :out:
        l: 1d tensor, VAE loss
        """
        return F.mse_loss(recon_x, x)

    def initialize(self):
        """
        Model Initialization
        """
        def _init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

        self.model.apply(_init_weights)
        return