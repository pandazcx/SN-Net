import torch
import torch.nn as nn
import math

class newton_modify(nn.Module):
    def __init__(self,args):
        super(newton_modify,self).__init__()
        self.N_1 = args["N_1"]
        self.N_2 = args["N_2"]
        self.M = 3 * args["M"]
        self.con_num = args["con_num"]
        self.sigma_0 = torch.ones(1,self.M,1,1).cuda()
        self.filter_size = 3
        self.c = nn.Parameter(torch.full([self.N_1,self.M],1.,device="cuda"),requires_grad=True)
        self.eta_ = nn.Parameter(torch.full([self.N_1, self.N_2], 0.001, device="cuda"), requires_grad=True) #0.001
        self.weight_a = nn.Parameter(0.01 * nn.init.xavier_normal_(torch.empty(self.N_1, 2, self.M, self.con_num, device="cuda")))
        self.weight_b = nn.Parameter(0.01 * nn.init.xavier_normal_(torch.empty(self.N_1, 2, self.M, self.con_num, device="cuda")))
        self.bias = nn.Parameter(0.01 * nn.init.xavier_normal_(torch.empty(self.N_1, 2,self.M, device="cuda")))
        self.init_conv = nn.Conv2d(in_channels=3, out_channels=self.M,groups=3,
                                   kernel_size=3,padding="same",padding_mode="circular",stride=1,bias=False)
        layers = []
        for i in range(self.N_1):
            layers.append(Img_update(self.weight_a[i],self.weight_b[i],self.bias[i],self.N_2,self.filter_size,self.M,self.c[i],self.eta_[i],True))

        self.net = nn.Sequential(*layers)

    def forward(self,x):
        '''
        input x: B,1,H,W ; k_z: B,1,H,W ; k_k: H,W ; lambda_out: B,C,H,W ; sigma:1,C,1,1 ; f_f: C,H,W ;

        '''
        Iteration = dict()
        Iteration['z'] = x #B,C,H,W
        Iteration['u_out'] = x # B,C,H,W
        Iteration['lambda_out'] = self.init_conv(x)
        Iteration['sigma'] = self.sigma_0 #1,C,1,1
        out = self.net(Iteration)
        #out_img = out['u_out']
        out_img = torch.clip(out['u_out'],0.0,1.0)
        return out_img

class Conv_mimc_1(nn.Module):
    def __init__(self,in_channel,out_channel,ksize=3):
        super(Conv_mimc_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel*2, groups=1, kernel_size=ksize,
                               padding="same", padding_mode="circular", bias=False)
        self.conv2 = nn.Conv2d(in_channels=in_channel*3, out_channels=in_channel*3, groups=3, kernel_size=ksize,
                               padding="same",
                               padding_mode="circular", bias=False)
        self.conv3 = nn.Conv2d(in_channels=in_channel*6, out_channels=out_channel, groups=6,
                               kernel_size=ksize, padding="same",
                               padding_mode="circular", bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3

class Conv_mimc_2(nn.Module):
    def __init__(self,in_channel,out_channel,ksize=3):
        super(Conv_mimc_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=4, groups=4, kernel_size=ksize,
                               padding="same", padding_mode="circular", bias=False)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, groups=8, kernel_size=ksize,
                               padding="same",
                               padding_mode="circular", bias=False)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=12, groups=12,
                               kernel_size=ksize, padding="same",
                               padding_mode="circular", bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3

class Img_update(nn.Module):
    def __init__(self,weight_a,weight_b, bias, N_2,filter_size,M,c,eta_,is_rgb):
        super(Img_update,self).__init__()
        self.weight_a = weight_a #2,M,con
        self.weight_b = weight_b #2,M,con
        self.bias = bias #2,M
        self.channel_num = M
        self.eta_ = eta_
        self.is_rgb = is_rgb

        self.N_2 = N_2
        self.c = c
        self.con_num = weight_a.size()[-1]
        self.freq = torch.arange(1, self.con_num+1, device="cuda") * math.pi  # C,con
        self.layer_simc = nn.ModuleList([Conv_mimc_1(in_channel=3,out_channel=M,ksize=filter_size) for i in range(self.N_2 + 2)])
        self.layer_mimc = nn.ModuleList([Conv_mimc_2(in_channel=M,out_channel=M,ksize=filter_size) for i in range(self.N_2+1)])

    def pwl(self,x):
        '''
        :param x: B,C,H,W
        self.control_b: C,101
        :return: B,C,H,W
        '''
        x_p = x.expand(self.con_num,x.shape[0],x.shape[1],x.shape[2],x.shape[3]).permute(1,2,3,4,0) #B,C,H,W,con
        weight_a = self.weight_a.expand(x.shape[0], x.shape[2], x.shape[3], 2, self.weight_a.shape[1], self.weight_a.shape[2]).permute(3,0,4,1,2,5) #2,B,C,H,W,con
        weight_b = self.weight_b.expand(x.shape[0], x.shape[2], x.shape[3], 2, self.weight_b.shape[1], self.weight_b.shape[2]).permute(3,0,4,1,2,5) #2,B,C,H,W,con
        Fourier_sin = torch.sin(self.freq * x_p) #B,C,H,W,con
        Fourier_cos = torch.cos(self.freq * x_p) #B,C,H,W,con
        bias_p = self.bias.expand(x.shape[0], x.shape[2], x.shape[3], 2 , self.bias.shape[1]).permute(3,0,4,1,2) #2,B,C,H,W
        out = torch.sum(weight_a[0] * Fourier_sin + weight_b[0] * Fourier_cos,dim=-1) + bias_p[0]
        sd = torch.sum(weight_a[1] * Fourier_sin + weight_b[1] * Fourier_cos,dim=-1) + bias_p[1]
        return out,sd

    def com_fun(self,u_in,z,lambda_in,sigma,epoch):
        f_u = self.layer_simc[-1](u_in)
        th, dri = self.pwl(lambda_in / sigma + f_u)
        out = self.layer_mimc[-1](lambda_in) + sigma * self.layer_mimc[-1](f_u - th)
        cm = int(self.channel_num / 3)
        if self.is_rgb:
            mid_r = torch.sum(out[:,:cm,:,:], dim=1).unsqueeze(1)
            mid_g = torch.sum(out[:,cm:2*cm,:,:],dim=1).unsqueeze(1)
            mid_b = torch.sum(out[:,2*cm:,:,:],dim=1).unsqueeze(1)
            out = torch.cat((mid_r,mid_g,mid_b),dim=1)+u_in-z
        else:
            out = torch.sum(out, dim=1).unsqueeze(1)+u_in - z

        f_f_o = self.layer_simc[epoch](out)
        f_f_o = self.layer_mimc[epoch](f_f_o * (torch.ones_like(dri) - dri))

        if self.is_rgb:
            gmid_r = torch.sum((sigma * f_f_o)[:,:cm,:,:], dim=1).unsqueeze(1)
            gmid_g = torch.sum((sigma * f_f_o)[:,cm:2*cm,:,:], dim=1).unsqueeze(1)
            gmid_b = torch.sum((sigma * f_f_o)[:,2*cm:,:,:], dim=1).unsqueeze(1)
            grad = torch.cat((gmid_r,gmid_g,gmid_b),dim=1)+out
        else:
            grad = out + torch.sum((sigma * f_f_o), dim=1).unsqueeze(1)
        return out,grad

    def solve_fun(self,u_in,z,lambda_in,sigma):
        for epoch in range(self.N_2): #5
            fun,grad = self.com_fun(u_in,z,lambda_in,sigma,epoch)
            rate = self.eta_[epoch]
            u_in = u_in -rate*grad
        return u_in

    def forward(self,x):
        u_in = x['u_out']
        lambda_in = x['lambda_out']
        sigma = x['sigma']
        z = x['z']
        u_out = self.solve_fun(u_in,z,lambda_in,sigma)
        f_o = self.layer_simc[-1](u_out)
        p_out, _ = self.pwl((lambda_in / sigma + f_o))
        x['u_out'] = u_out
        x['lambda_out'] = lambda_in + sigma * (self.layer_simc[-2](u_out) - p_out)
        x['sigma'] = (self.c).unsqueeze(1).unsqueeze(1).unsqueeze(0)*sigma

        return x


