import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from models import BaseModel
from models.utils import *

class Exclusive_Specific_Encoder(BaseModel):

    def __init__(self):
        super(Exclusive_Specific_Encoder, self).__init__()

        # input: Batch_size * 3 * 256 * 256
        self.layer_dims = [32, 64, 128, 256]
        self.mu_dims = 64 # previously 8
        self.latent_dims = self.mu_dims * 2
        
        # in: B * 20000
        # layer 1 out: B * 32 
        # layer 2 out: B * 64 
        # layer 3 out: B * 128 
        # layer 4 out: B * 256 

        downsample = []
        in_channels = 20000
        for out_channels in self.layer_dims: 
            downsample.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.02)
                )
            )
            in_channels = out_channels
        self.downsample = nn.Sequential(*downsample)
        self.fc_mu_var = nn.Sequential(nn.Linear(self.layer_dims[-1], self.latent_dims))

        self.weight_init()

        
    def forward(self, input):
        B, _, = input.size()
        out = self.downsample(input)
        out = out.view(B, -1)
        out = self.fc_mu_var(out)
        mu, log_var = torch.split(out, [self.mu_dims, self.mu_dims], dim = -1)
        z = self.reparameterize(mu, log_var)
        # print(mu)
        # print(mu.size())
        # print(log_var.size())
        # print(z.size())
        return [mu, log_var, z]

class Shared_Feature_extractor(BaseModel):
    def __init__(self):
        super(Shared_Feature_extractor, self).__init__()
        self.layer_dims = [32, 64, 128, 256]
        downsample = []
        in_channels = 20000
        for out_channels in self.layer_dims: 
            downsample.append(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.02)
                )
            )
            in_channels = out_channels
        self.downsample = nn.Sequential(*downsample)
        self.weight_init()


    def forward(self, input):
        out = self.downsample(input)
        return out

class Exclusive_Shared_Encoder(BaseModel):
    def __init__(self):
        super(Exclusive_Shared_Encoder, self).__init__()

        self.mu_dims = 128
        self.latent_dims = self.mu_dims * 2

        downsample = []
        in_channels = 256
        out_channels = 256
        downsample.append(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.02)
            )
        )
        self.downsample = nn.Sequential(*downsample)
        self.fc_mu_var = nn.Sequential(
            nn.Linear(out_channels, self.latent_dims) , 
            nn.Linear(self.latent_dims, self.latent_dims)
        )

        self.weight_init()


    def forward(self, input):
        B, _ = input.size()
        out = self.downsample(input)
        out = out.view(B, -1)
        out = self.fc_mu_var(out)
        mu, log_var = torch.split(out, [self.mu_dims, self.mu_dims], dim = -1)
        z = self.reparameterize(mu, log_var)
        return [mu, log_var, z]


class Common_Shared_Encoder(BaseModel):
    
    def __init__(self):
        super(Common_Shared_Encoder, self).__init__()
        self.mu_dims = 128
        self.latent_dims = self.mu_dims * 2

        downsample = []
        in_channels = 256 * 2
        out_channels = 256
        downsample.append(
            nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.02)
            )
        )
        self.downsample = nn.Sequential(*downsample)
        self.fc_mu_var = nn.Sequential(
            nn.Linear(out_channels, self.latent_dims),
            nn.Linear(self.latent_dims, self.latent_dims)
        )

        self.weight_init()


    def forward(self, inputX, inputY):
        input = torch.cat((inputX, inputY), dim = 1) #channel-wise concat
        B, _ = input.size()
        out = self.downsample(input)
        out = out.view(B, -1)
        out = self.fc_mu_var(out)
        mu, log_var = torch.split(out, [self.mu_dims, self.mu_dims], dim = -1)
        z = self.reparameterize(mu, log_var)

        return [mu, log_var, z]


