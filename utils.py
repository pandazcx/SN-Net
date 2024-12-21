import os
import sys
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import torch
import math
import torch.nn.functional as F
import torchvision.transforms as transforms

def batch_conv(img,k):
    Hi,Wi = img.shape[-2:]
    Hk,Wk = k.shape[-2:]
    H = Hi + 2*Hk - 2
    W = Wi + 2*Wk - 2
    Ho = Hi + Hk -1
    Wo = Wi + Wk -1
    IMG = torch.fft.fft2(img,s=(H,W))
    K = torch.fft.fft2(k,s=(H,W))
    CONV = plural_mul(IMG,K)
    conv = torch.fft.ifft2(CONV)[:,:,Hk//2:Ho-Hk//2,Wk//2:Wo-Wk//2].real
    return conv
def batch_ssim(ori_img,detect_img):
    #B,C,H,W
    if ori_img.size()[0] != detect_img.size()[0]:
        print("pinmode wrong")
        sys.exit(0)
    else:
        batchsize = ori_img.size()[0]
        ssim = 0
        for i in range(batchsize):
            ssim += structural_similarity(ori_img[i].data.cpu().numpy(), detect_img[i].data.cpu().numpy(),channel_axis=0, win_size=11, gaussian_weights=True, multichannel=False,
                                     data_range=1., K1=0.01, K2=0.03, sigma=1.5)
        return (ssim/batchsize)

def batch_PSNR(ori_img, detect_img):
    if ori_img.shape[0] != detect_img.shape[0]:
        print("pinmode wrong")
        sys.exit(0)
    ori_img = ori_img.data.cpu().numpy()
    detect_img = detect_img.data.cpu().numpy()
    PSNR = 0
    batchsize = ori_img.shape[0]
    for i in range(batchsize):
        PSNR += peak_signal_noise_ratio(ori_img[i], detect_img[i], data_range=1.)
    return (PSNR/batchsize)

def Matmul(x,p,q):

    out = torch.matmul(torch.matmul(p.softmax(dim=-1), x), q.softmax(dim=-2))
    return out

# def trans_Matmul(x,p,q):
#     pt = torch.transpose(p, -1, -2)
#     qt = torch.transpose(q, -1, -2)
#     out = torch.matmul(torch.matmul(pt.softmax(dim=-1), x), qt.softmax(dim=-2))
#     return out

def padding(scale,img):
    _,_,H,W = img.shape
    out_H = math.ceil(H / scale) * scale
    out_W = math.ceil(W / scale) * scale
    pad_H = out_H - H
    pad_W = out_W - W
    pad = (pad_W // 2, pad_W - (pad_W // 2),pad_H // 2, pad_H - (pad_H // 2))
    pad_img = F.pad(img,pad,mode="reflect")
    return pad_img,pad

def inv_padding(pad,img):
    _, _, H, W = img.shape
    out_img = img[:,:,pad[2]:H-pad[3],pad[0]:W-pad[1]]
    return out_img

def plural_div(a,b):
    real = a.real * b.real + a.imag * b.imag
    norm = b.real * b.real + b.imag * b.imag
    imag = a.imag * b.real - a.real * b.imag
    result = torch.complex(real,imag)
    result = result / norm
    return result

def plural_mul(a,b):
    real = a.real * b.real - a.imag * b.imag
    imag = a.real * b.imag + a.imag * b.real
    out = torch.complex(real,imag)
    return out

def conj(a):
    real = a.real
    imag = -a.imag
    return torch.complex(real,imag)

def modulus(a):
    real = a.real
    imag = a.imag
    result = real*real + imag*imag
    return result

def pad_to(original, size):
    '''
    Post-pad last two dimensions to "size"
    '''

    original_size = original.size()
    pad = [0, size[1] - original_size[-1],
           0, size[0] - original_size[-2]]

    return F.pad(original, pad)


def fft2(signal, size=None):
    '''
    Fast Fourier transform on the last two dimensions
    '''

    padded = signal if size is None else pad_to(signal, size)
    return torch.fft.fft2(padded)

def image_save(img,dirpath,name):
    topil = transforms.ToPILImage()
    img = topil(img)
    path = os.path.join(dirpath,name)
    img.save(path)


#=============================
def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]

def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = torch.zeros([B, C, H, W],device=windows.device)
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res

