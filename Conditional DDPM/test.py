from evaluator import evaluation_model
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import plot_images,save_images,setup_logging
from model import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from dataloader_test import testDataset
from dataloader import iclevrDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from PIL import Image
import torchvision.transforms as transforms
# In[]
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=4):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
# In[]
def sort_by_number(file_name):
    number = int(''.join(filter(str.isdigit, file_name)))
    return number

# In[]
def test(args):
    setup_logging(args.run_name)
    device = args.device

    dataloader=iclevrDataset(mode= args.mode, root='dataset')

    dataloader = DataLoader(dataloader, batch_size=args.batch_size, shuffle=False)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    diffusion = Diffusion(img_size=args.image_size, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt")))
    ema_model.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ema_ckpt.pt")))
    optimizer.load_state_dict(torch.load(os.path.join("models", args.run_name, f"optim.pt")))
    #生成影像，存到dataset/test
    for i,labels in enumerate(dataloader):
        labels=labels.to(torch.float32).to(device)
        sampled_images = diffusion.sample(model, n=1, labels=labels)
        plot_images(sampled_images)
        save_images(sampled_images, os.path.join("dataset/new_test", f"{i}.jpg"))
        
    #讀取生成的影像放到images，並做成grid
    folder_path='dataset/new_test'
    file_names = sorted(os.listdir(folder_path),key=sort_by_number)
    transform_test = transforms.ToTensor()
    images = []
    for file_name in file_names:
        print(file_name)

        file_path = os.path.join(folder_path, file_name)
        
        image = Image.open(file_path)
        
        tensor_image = transform_test(image)
        
        images.append(tensor_image)

    grid_image = make_grid(images, nrow=8)
    save_image(grid_image, "grid_imageema.jpg")
    transform_norm = transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])
    datatest=testDataset(mode=args.mode,root='dataset') #得到one hot
    datatest = DataLoader(datatest, batch_size=1, shuffle=False)
    evaluator = evaluation_model()
    with torch.no_grad():
        final_acc =0
        for i, cond in enumerate(datatest):
            cond = cond.to(args.device)
            img=transform_norm(images[i]).unsqueeze(0).to(args.device)
            acc = evaluator.eval(img, cond)
            print(acc)
            final_acc += acc
        print(f'Average acc: {final_acc*100/32:.2f}%')
    
    
    
# In[]
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.batch_size = 1
    args.image_size = 64
    args.num_classes = 141
    args.lr = 3e-4
    args.device = "cuda"
    args.mode='new_test'
    test(args)
# In[]
launch()