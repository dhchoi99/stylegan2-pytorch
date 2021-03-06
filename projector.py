import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]
        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2
    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .numpy()
    )


class Projector:
    def __init__(self, args):
        self.args = args

        self.init_vars()

        self.percept = lpips.PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=self.args.device.startswith('cuda')
        )

    # TODO
    def init_vars(self):
        self.args.device = 'cuda'
        self.args.size = 512
        self.args.resize = min(self.args.size, 256)
        self.args.steps = 1000
        self.args.n_mean_latent = 10000

        self.args.lr_rampup = 0.05
        self.args.lr_rampdown = 0.25
        self.args.lr = 0.1

        self.args.noise = 0.05
        self.args.noise_ramp = 0.75
        self.args.noise_regularize = 1e5

        self.args.mse = 0
        self.args.w_plus = False

        self.transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.CenterCrop(self.args.resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def read_img(self, img):
        if img is not None:
            if isinstance(img, str):
                img = self.transform(Image.open(img).convert('RGB'))
            # elif isinstance(img, PIL.Image):
            #     img = self.transform(img)
            elif isinstance(img, torch.Tensor):
                pass
        return img


    def project(self, generator, img, steps=1000):
        img = self.read_img(img)
        img = img.to(self.args.device)

        with torch.no_grad():
            noise_sample = torch.randn(self.args.n_mean_latent, 512, device=self.args.device)
            latent_out = generator.style(noise_sample)
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / self.args.n_mean_latent) ** 0.5

        noises_single = generator.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.normal_())

        latent_in = latent_mean.clone().detach().unsqueeze(0)
        if self.args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, generator.n_latent, 1)
        latent_in.requires_grad_(True)

        for noise in noises:
            noise.requires_grad_(True)

        optimizer = optim.Adam([latent_in] + noises, lr=self.args.lr)
        latent_path = []

        pbar = tqdm(range(steps))

        for i in pbar:
            t = i / steps
            lr = get_lr(t, self.args.lr)

            optimizer.param_groups[0]['lr'] = lr
            noise_strength = latent_std * self.args.noise * max(0, 1-t/self.args.noise_ramp)**2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = generator([latent_n], input_is='latent', noise=noises)

            B, C, H, W = img_gen.shape
            if H > self.args.resize:
                factor = H // self.args.resize

                img_gen = img_gen.reshape(B, C, H // factor, factor, W // factor, factor)
                img_gen = img_gen.mean([3, 5])

            p_loss = self.percept(img_gen, img).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, img)

            loss = p_loss + self.args.noise_regularize * n_loss + self.args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.clone().detach())

            pbar.set_description((
                f"perceptual: {p_loss.item():.8f}; noise regularize: {n_loss.item():.8f}; mse: {mse_loss.item():.8f}; lr: {lr:.4f}"
            ))

        img_gen, _ = generator(latent_path[-1], input_is='latent', noise=noises)
        return img_gen, latent_path, latent_in


def project(path_ckpt, path_files, step=1000):
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='jup kernel')
    # parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--noise_regularize", type=float, default=1e5)
    parser.add_argument("--mse", type=float, default=0)
    # parser.add_argument("--w_plus", action="store_true")
    # parser.add_argument("files", metavar="FILES", nargs="+")

    args = parser.parse_args()
    args.ckpt = path_ckpt
    args.files = path_files
    args.w_plus = False
    args.step = step

    n_mean_latent = 10000
    resize = min(args.size, 256)
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    imgs = []
    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)
    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen, _ = g_ema([latent_n], input_is='latent', noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > resize:
            factor = height // resize

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description((
            f"perceptual: {p_loss.item():.8f}; noise regularize: {n_loss.item():.8f}; mse: {mse_loss.item():.8f}; lr: {lr:.4f}"
        ))

    img_gen, _ = g_ema([latent_path[-1]], input_is='latent', noise=noises)

    filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"

    img_ar = make_image(img_gen)

    result_file = {}
    for i, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i: i + 1])

        result_file[input_name] = {
            "img": img_gen[i],
            "latent": latent_in[i],
            "noise": noise_single,
        }

        img_name = os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)

    torch.save(result_file, filename)
    print(filename)

    return img_gen, latent_path, latent_in


if __name__ == "__main__":
    project()
