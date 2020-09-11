import argparse, math, random, os

import numpy as np
import tarfile
import torch
from torch import nn, autograd, optim
import torch.nn.functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None
    from torch.utils.tensorboard import SummaryWriter

from model import Generator, Discriminator
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_augment_data = torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment += reduce_sum(ada_augment_data)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        f"sample/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )


def calc_loss_1(args, generator, discriminator, real_img, ada_aug_p):
    losses = {}
    noise = mixing_noise(args.batch, args.latent, args.mixing, args.device)
    fake_img, _ = generator(noise)

    if args.augment:
        real_img_aug, _ = augment(real_img, ada_aug_p)
        fake_img, _ = augment(fake_img, ada_aug_p)
    else:
        real_img_aug = real_img

    fake_pred = discriminator(fake_img)
    real_pred = discriminator(real_img_aug)
    d_loss = d_logistic_loss(real_pred, fake_pred)

    losses['d'] = d_loss
    losses['real_score'] = real_pred.mean()
    losses['fake_score'] = fake_pred.mean()

    return losses, real_pred


def calc_loss_2(discriminator, real_img):
    losses = {}

    real_img.requires_grad = True
    real_pred = discriminator(real_img)
    r1_loss = d_r1_loss(real_pred, real_img)

    losses['r1'] = r1_loss
    return losses, real_pred


def calc_loss_3(args, generator, discriminator, ada_aug_p):
    losses = {}

    noise = mixing_noise(args.batch, args.latent, args.mixing, args.device)
    fake_img, _ = generator(noise)

    if args.augment:
        fake_img, _ = augment(fake_img, ada_aug_p)

    fake_pred = discriminator(fake_img)
    g_loss = g_nonsaturating_loss(fake_pred)
    losses['g'] = g_loss

    return losses


def calc_loss_4(args, generator, device, mean_path_length):
    losses = {}

    path_batch_size = max(1, args.batch // args.path_batch_shrink)
    noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
    fake_img, latents = generator(noise, return_latents=True)

    path_loss, mean_path_length, path_lengths = g_path_regularize(
        fake_img, latents, mean_path_length
    )
    weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

    if args.path_batch_shrink:
        weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

    losses['path'] = path_loss
    losses['path_length'] = path_lengths.mean()
    losses['weighted_path'] = weighted_path_loss

    return losses, mean_path_length


def train2(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, logger):
    loader = sample_data(loader)

    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # init vars
    mean_path_length = 0
    mean_path_length_avg = 0
    losses = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    ada_aug_step = args.ada_target / args.ada_length
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        loss, real_pred = calc_loss_1(args, generator, discriminator, real_img, ada_aug_p)
        d_loss = loss['d']

        discriminator.zero_grad()  #
        d_loss.backward()
        d_optim.step()
        losses.update(loss)

        #

        if args.augment and args.augment_p == 0:
            ada_augment_data = torch.tensor(
                (torch.sign(real_pred).sum().item(), real_pred.shape[0]), device=device
            )
            ada_augment += reduce_sum(ada_augment_data)

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target:
                    sign = 1

                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        # loss2
        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            loss, real_pred = calc_loss_2(discriminator, real_img)
            r1_loss = loss['r1']

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()
            losses.update(loss)

        # loss 3
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        loss = calc_loss_3(args, generator, discriminator, ada_aug_p)
        g_loss = loss['g']
        losses.update(loss)

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0
        if g_regularize:
            loss, mean_path_length = calc_loss_4(args, generator, device, mean_path_length)

            loss['weighted_path'].backward()
            g_optim.step()

            mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())
            losses.update(loss)

        accumulate(g_ema, g_module, accum)
        loss_reduced = reduce_loss_dict(losses)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description((
                f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                f"augment: {ada_aug_p:.4f}"
            ))

            # write log
            write_log(loss_reduced, 'train', i, logger)
            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        f"sample/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )


def write_log(losses, mode, step, logger=None):
    assert isinstance(losses, dict)
    assert isinstance(logger, SummaryWriter)

    for key, value in losses.items():
        logger.add_scalar('%s/%s' % (mode, key), value.item(), step)


class StyleGAN2:
    def __init__(self):
        super(StyleGAN2, self).__init__()

    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--path", type=str)
        parser.add_argument('--start_iter', type=int, default=0)
        parser.add_argument("--iter", type=int, default=800000)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--n_sample", type=int, default=64)
        parser.add_argument("--size", type=int, default=256)
        parser.add_argument("--r1", type=float, default=10)
        parser.add_argument("--path_regularize", type=float, default=2)
        parser.add_argument("--path_batch_shrink", type=int, default=2)
        parser.add_argument("--d_reg_every", type=int, default=16)
        parser.add_argument("--g_reg_every", type=int, default=4)
        parser.add_argument("--mixing", type=float, default=0.9)
        parser.add_argument("--ckpt", type=str, default=None)
        parser.add_argument("--lr", type=float, default=0.002)
        parser.add_argument('--latent', type=int, default=512)
        parser.add_argument('--n_mlp', type=int, default=8)
        parser.add_argument("--channel_multiplier", type=int, default=2)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--log_dir', type=str, default='log')
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--augment", action="store_true")
        parser.add_argument("--augment_p", type=float, default=0)
        parser.add_argument("--ada_target", type=float, default=0.6)
        parser.add_argument("--ada_length", type=int, default=500 * 1000)

        args = parser.parse_args()
        self.args = args
        return args

    def set_vars(self):
        self.parse_args()

        self.args.manualSeed = 1

        self.args.n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.args.distributed = self.args.n_gpu > 1

        if self.args.distributed:
            torch.cuda.set_device(self.args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

    def set_nets(self, mode=None):
        args = self.args
        self.generator = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        self.discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        self.g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(args.device)
        self.g_ema.eval()
        accumulate(self.g_ema, self.generator, 0)

    def set_optims(self):
        args = self.args
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        self.d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

    def load_weight(self):
        args = self.args
        if args.ckpt is not None:
            print("load model:", args.ckpt)

            ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

            try:
                ckpt_name = os.path.basename(args.ckpt)
                args.start_iter = int(os.path.splitext(ckpt_name)[0])

            except ValueError:
                pass

            try:
                self.generator.load_state_dict(ckpt["g"])
                self.discriminator.load_state_dict(ckpt["d"])
                self.g_ema.load_state_dict(ckpt["g_ema"])
                try:
                    self.g_optim.load_state_dict(ckpt["g_optim"])
                    self.d_optim.load_state_dict(ckpt["d_optim"])
                except:
                    print('loading optim failed')
            except KeyError:
                self.generator.load_state_dict(ckpt)

        if args.distributed:
            self.generator = nn.parallel.DistributedDataParallel(
                self.generator,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

    def set_dataset(self):
        args = self.args
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])

        self.dataset = MultiResolutionDataset(args.path, transform, args.size)
        self.loader = data.DataLoader(
            self.dataset,
            batch_size=args.batch,
            sampler=data_sampler(self.dataset, shuffle=True, distributed=args.distributed),
            drop_last=True,
        )

    def set_logger(self):
        if get_rank() == 0 and wandb is not None and self.args.wandb:
            wandb.init(project="stylegan 2")
        else:
            self.log_dir = '%s/%d' % (self.args.log_dir, self.args.manualSeed)
            os.makedirs(self.log_dir, exist_ok=True)
            self.summary = SummaryWriter(log_dir=self.log_dir)

        with tarfile.open(os.path.join(self.log_dir, 'code.tar.gz'), "w:gz") as tar:
            for addfile in ['train.py', 'dataset.py', 'model.py']:
                tar.add(addfile)
        '''with open(os.path.join(self.log_dir, 'args.txt'), 'w') as f:
            for key, value in dict(self.args).items():
                s = key + '\t\t' + value + '\n'
                f.write(s)'''

    def train(self):
        self.set_vars()
        self.args.cur_iter = 0
        self.set_dataset()
        self.set_nets()
        self.set_optims()
        self.load_weight()
        self.set_logger()

        train2(
            self.args, self.loader, self.generator, self.discriminator,
            self.g_optim, self.d_optim, self.g_ema, self.args.device, self.summary)


if __name__ == "__main__":
    s = StyleGAN2()
    s.train()
