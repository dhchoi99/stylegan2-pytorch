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
    def __init__(self):
        self.init_vars()
        self.set_network()

    def init_vars(self):
        self.size = 512
        self.lr_rampup = 0.05
        self.lr_rampdown = 0.25
        self.lr = 0.1
        self.noise = 0.05
        self.noise_ramp = 0.75
        self.step = 1000
        self.noise_regularize = 1e5
        self.mse = 0
        self.w_plus = True
        self.n_mean_latent = 10000
        self.device = 'cuda'

        self.verbose = True
        self.resize = min(self.size, 256)

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_network(self):
        path = '../stylegan2/checkpoint/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pt'
        self.g_ema = Generator(self.size, 512, 8)
        self.g_ema.load_state_dict(torch.load(path)['g_ema'], strict=False)
        self.g_ema.eval()
        self.g_ema.to(self.device)

        # Find dlatent stats.
        self._info('Finding W midpoint and stddev using %d samples' % self.n_mean_latent)
        with torch.no_grad():
            noise_sample = torch.randn(self.n_mean_latent, self.size, device=self.device)
            latent_out = self.g_ema.style(noise_sample)

            self.latent_mean = latent_out.mean(0)
            self.latent_std = ((latent_out - self.latent_mean).pow(2).sum() / self.n_mean_latent) ** 0.5

        self.percept = lpips.PerceptualLoss(
            model='net-lin', net='vgg', use_gpu=self.device.startswith('cuda')
        )

        # Find noise inputs
        self._info('Setting up noise inputs...')
        noises_single = self.g_ema.make_noise()
        self.noises = []
        rep = 3
        for noise in noises_single:
            self.noises.append(noise.repeat(rep, 1, 1, 1).normal_())  # TODO
        self.latent_in = self.latent_mean.detach().clone().unsqueeze(0).repeat(rep, 1)

        if self.w_plus:
            self.latent_in = self.latent_in.unsqueeze(1).repeat(1, self.g_ema.n_latent, 1)
        self.latent_in.requires_grad_(True)
        for noise in self.noises:
            noise.requires_grad_(True)

        self.optimizer = optim.Adam([self.latent_in] + self.noises, lr=self.lr)

    def optimize(self, imgs):
        pbar = tqdm(range(self.step))
        latent_path = []

        for i in pbar:
            t = i / self.step
            lr = get_lr(t, self.lr)
            self.optimizer.param_groups[0]['lr'] = lr
            noise_strength = self.latent_std * self.noise * max(0, 1 - t / self.noise_ramp) ** 2
            latent_n = latent_noise(self.latent_in, noise_strength.item())

            img_gen, _ = self.g_ema([latent_n], input_is_latent=True, noise=self.noises)
            B, C, H, W = img_gen.shape

            if H > 512:
                factor = H // 512

                img_gen = img_gen.reshape(
                    B, C, H // factor, W // factor, factor
                )
                img_gen = img_gen.mean([3, 5])  # TODO what is this for?

            p_loss = self.percept(img_gen, imgs).sum()
            n_loss = noise_regularize(self.noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            loss = p_loss + self.noise_regularize * n_loss + self.mse * mse_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            noise_normalize_(self.noises)

            if (i + 1) % 100 == 0:
                latent_path.append(self.latent_in.detach().clone())

            pbar.set_description((
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.8f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            ))

        img_gen, _ = self.g_ema([latent_path[-1]], input_is_latent=True, noise=self.noises)
        img_ar = make_image(img_gen)

        return (latent_path, self.noises), (img_gen, img_ar)


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

        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

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

    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

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


class Projector2:
    def __init__(self):
        self.init_vars()

    def init_vars(self):
        self.num_steps                  = 1000
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.1
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5
        self.verbose                    = False
        self.clone_net                  = True
        self.device                     = 'cuda'

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._noise_in              = None
        self._dlatents_expr         = None
        self._images_expr           = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None

    def _info(self, *args):
        if self.verbose:
            print('Projector:', *args)

    def set_network(self, torch_Gs, minibatch_size=1):
        assert minibatch_size == 1
        self.torch_Gs = Generator(512, 512, 8)
        self.torch_Gs.load_state_dict(torch_Gs.state_dict())
        self.torch_Gs.to(self.device)
        self._minibatch_size = minibatch_size

        # Find dlatent stats
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        latent_samples = torch.randn(self.dlatent_avg_samples, 512, device=self.device)
        dlatent_samples = self.torch_Gs.style(latent_samples)
        self._dlatent_avg = dlatent_samples.mean(0)
        self._dlatent_std = ((dlatent_samples-self._dlatent_avg).pow(2).sum() / self.dlatent_avg_samples) ** 0.5
        self._info('std = %g' % self._dlatent_std)

        # Find noise inputs.
        self._info('Setting up noise inputs...')
        #self._noise_vars = []
        self._noise_vars = self.torch_Gs.make_noise()
        noise_init_ops = []
        noise_normalize_ops = []



if __name__ == "__main__":
    project()
