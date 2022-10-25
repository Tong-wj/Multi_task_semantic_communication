import numpy as np
import torch
import os
from .dynamic_base_model import BaseModel
from . import dynamic_networks
import math

class DynaJSCC(BaseModel):
    
    def __init__(self, cfg, backbone_channels):
        BaseModel.__init__(self, cfg)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'G_reward']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['SE', 'CE', 'G', 'P']

        # define dynamic_networks
        # self.netSE = dynamic_networks.define_SE(input_nc=cfg['input_nc'], ngf=cfg['ngf'], max_ngf=cfg['max_ngf'],
        #                                 n_downsample=cfg['n_downsample'], norm=cfg['norm'], init_type=cfg['init_type'],
        #                                 init_gain=cfg['init_gain'], gpu_ids=self.gpu_ids)
        self.netCE = dynamic_networks.define_CE(ngf=cfg['ngf'], max_ngf=cfg['max_ngf'], n_downsample=cfg['n_downsample'], C_channel=cfg['C_channel'],
                                        norm=cfg['norm'], init_type=cfg['init_type'],
                                        init_gain=cfg['init_gain'], gpu_ids=self.gpu_ids)

        self.netG = dynamic_networks.define_dynaG(output_nc=cfg['output_nc'], ngf=cfg['ngf'], max_ngf=cfg['max_ngf'],
                                          n_downsample=cfg['n_downsample'], C_channel=cfg['C_channel'],
                                          n_blocks=cfg['n_blocks'], norm=cfg['norm'], init_type=cfg['init_type'],
                                          init_gain=cfg['init_gain'], gpu_ids=self.gpu_ids)

        self.netP = dynamic_networks.define_dynaP(ngf=cfg['ngf'], max_ngf=cfg['max_ngf'],
                                          n_downsample=cfg['n_downsample'], init_type=cfg['init_type'],
                                          init_gain=cfg['init_gain'], gpu_ids=self.gpu_ids,
                                          N_output=cfg['G_s'] + 1)


        print('---------- dynamic_networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            # params = list(self.netSE.parameters()) + list(self.netCE.parameters()) + list(self.netG.parameters()) + list(self.netP.parameters())
            params = list(self.netCE.parameters()) + list(self.netG.parameters()) + list(self.netP.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=cfg['lr_joint'], betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)

        self.cfg= cfg
        self.temp = cfg['temp_init'] if cfg['isTrain'] else 5

    def name(self):
        return 'DynaJSCC_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def forward(self):
        # Generate SNR
        if self.cfg['isTrain']:
            self.snr = torch.rand(self.real_A.shape[0], 1).to(self.device) * (self.cfg['SNR_MAX']-self.cfg['SNR_MIN']) - self.cfg['SNR_MIN']
        else:
            self.snr = torch.ones(self.real_A.shape[0], 1).to(self.device) * self.cfg['SNR']

        # Generate latent vector
        # z = self.netSE(self.real_A)
        latent = self.netCE(self.real_A, self.snr)

        # Generate decision mask
        self.hard_mask, self.soft_mask, prob = self.netP(self.real_A, self.snr, self.temp)
        self.count = self.hard_mask.sum(-1)

        # Normalize each channel
        latent_sum = torch.sqrt((latent**2).mean((-2, -1), keepdim=True))
        latent = latent / latent_sum

        # Generate the full mask
        N, C, H, W = latent.shape
        self.cpp = (self.cfg['G_n'] + torch.mean(self.count))/(self.cfg['G_n'] + self.cfg['G_s']) * C * H * W / (3*480*640)

        pad = torch.ones((N, self.cfg['G_n']), device=self.device)
        self.hard_mask = torch.cat((pad, self.hard_mask), -1)
        self.soft_mask = torch.cat((pad, self.soft_mask), -1)
        
        latent_res = latent.view(N, self.cfg['G_s']+self.cfg['G_n'], -1)
        
        # Selection with either soft mask or hard mask
        if self.cfg['isTrain']:
            if self.cfg['select'] == 'soft':
                latent_res = latent_res * self.soft_mask.unsqueeze(-1)
            else:
                latent_res = latent_res * self.hard_mask.unsqueeze(-1)
        else:
            latent_res = latent_res * self.hard_mask.unsqueeze(-1)

        # Pass through the AWGN channel
        with torch.no_grad():
            sigma = 10**(-self.snr / 20)  
            noise = sigma.view(self.real_A.shape[0], 1, 1) * torch.randn_like(latent_res)
            if self.cfg['isTrain']:
                if self.cfg['select'] == 'soft':
                    noise = noise * self.soft_mask.unsqueeze(-1)
                else:
                    noise = noise * self.hard_mask.unsqueeze(-1)
            else:
                noise = noise * self.hard_mask.unsqueeze(-1)

        latent_res = latent_res + noise
        self.fake = self.netG(latent_res.view(latent.shape), self.snr)
        return self.fake

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B)
        self.loss_G_reward = torch.mean(self.count)
        self.loss_G = self.cfg['lambda_L2'] * self.loss_G_L2 + self.cfg['lambda_reward'] * self.loss_G_reward
        return self.loss_G, self.cpp
        # self.loss_G.backward()

    def optimize_parameters(self):

        self.forward()
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def update_temp(self):
        self.temp *= math.exp(-self.cfg['eta'])
        self.temp = max(self.temp, 0.005)
