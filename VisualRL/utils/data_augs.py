import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils.TransformLayer import ColorJitterLayer

def center_translate(imgs, size):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, :, h1:h1 + h, w1:w1 + w] = imgs
    return outs

def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs

def center_window(imgs, window_size, in_place=False):
    n, c, h, w = imgs.shape
    zeros_h = h - window_size
    zeros_w = w - window_size
    min_h = zeros_h // 2
    max_h = h - (zeros_h // 2)
    min_w = zeros_w // 2
    max_w = w - (zeros_w // 2)
    if max_w - min_w > window_size:
        max_w -= 1  # case of odd number
    if max_h - min_h > window_size:
        max_h -= 1  # case of odd number
    imgs = imgs if in_place else imgs.copy()
    imgs[:, :, :min_h, :] = 0.
    imgs[:, :, max_h:, :] = 0.
    imgs[:, :, :, :min_w] = 0.
    imgs[:, :, :, max_w:] = 0.
    return imgs
    
def random_window(imgs, window_size, in_place=False):
    n, c, h, w = imgs.shape
    zeros_h = h - window_size
    zeros_w = w - window_size
    min_h = np.random.randint(low=0, high=zeros_h + 1, size=n)
    min_w = np.random.randint(low=0, high=zeros_w + 1, size=n)
    max_h = min_h + window_size
    max_w = min_w + window_size
    imgs = imgs if in_place else imgs.copy()
    for img, mnh, mxh, mnw, mxw in zip(imgs, min_h, max_h, min_w, max_w):
        img[:, :mnh, :] = 0.
        img[:, mxh:, :] = 0.
        img[:, :, :mnw] = 0.
        img[:, :, mxw:] = 0.
    return imgs

def random_crop(imgs, out):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3
    
    imgs = imgs.view([b,frames,3,h,w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114 
    
    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(device) # broadcast tiling
    return imgs

def random_grayscale(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or cuda
        returns torch.tensor
    """
    device = images.device
    in_type = images.type()
    images = images * 255.
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.
    return out

# random cutout
# TODO: should mask this 

def random_cutout(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        min / max cut: int, min / max size of cutout 
        returns np.array
    """

    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        #print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
        cutouts[i] = cut_img
    return cutouts

def random_cutout_color(imgs, min_cut=10,max_cut=30):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """
    
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        
        # add random box
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = np.tile(
            rand_box[i].reshape(-1,1,1),                                                
            (1,) + cut_img[:, h11:h11 + h11, w11:w11 + w11].shape[1:])
        
        cutouts[i] = cut_img
    return cutouts

# random flip

def random_flip(images,p=.2):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or gpu, 
        p: prob of applying aug,
        returns torch.tensor
    """
    # images: [B, C, H, W]
    device = images.device
    bs, channels, h, w = images.shape
    
    images = images.to(device)

    flipped_images = images.flip([3])
    
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] #// 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None]
    
    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, -1, h, w])
    return out

# random rotation

def random_rotation(images,p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: str, cpu or gpu, 
        p: float, prob of applying aug,
        returns torch.tensor
    """
    device = images.device
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    
    images = images.to(device)

    rot90_images = images.rot90(1,[2,3])
    rot180_images = images.rot90(2,[2,3])
    rot270_images = images.rot90(3,[2,3])    
    
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)
    
    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i,m in enumerate(masks):
        m[torch.where(mask==i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(device)
        m = m[:,:,None,None]
        masks[i] = m
    
    
    out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    out = out.view([bs, -1, h, w])
    return out


# random color

def random_convolution(imgs):
    '''
    random covolution in "network randomization"
    
    (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
    '''
    _device = imgs.device
    
    img_h, img_w = imgs.shape[2], imgs.shape[3]
    num_stack_channel = imgs.shape[1]
    num_batch = imgs.shape[0]
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)
    
    # initialize random covolution
    rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
    
    for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        temp_imgs = imgs[trans_index*batch_size:(trans_index+1)*batch_size]
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
    return total_out


def random_color_jitter(imgs):
    """
        inputs np array outputs tensor
    """
    b,c,h,w = imgs.shape
    imgs = imgs.view(-1,3,h,w)
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4, 
                                                contrast=0.4,
                                                saturation=0.4, 
                                                hue=0.5, 
                                                p=1.0, 
                                                batch_size=128))

    imgs = transform_module(imgs).view(b,c,h,w)
    return imgs

def no_aug(x):
    return x

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image

def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image

