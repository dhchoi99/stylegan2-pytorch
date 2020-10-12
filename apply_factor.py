import argparse
import torch
from torchvision import utils
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from model import Generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", type=int, default=0)
    parser.add_argument("-d", "--degree", type=float, default=5, help='max degree')
    parser.add_argument('--m', type=int, default=3, help='imgs per latent')
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("-n", "--n_sample", type=int, default=7)
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_prefix", type=str, default="factor")
    parser.add_argument("--factor", type=str)
    args = parser.parse_args()
    assert args.m >= 0

    return args


def compare_axes():
    torch.set_grad_enabled(False)
    device = 'cuda'
    path_factor = 'factor.pt'
    m = 3
    degree = 10
    truncation = 0.7
    out_prefix = 'copmare_axes'
    imsize = 256

    # eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    f_dict = torch.load(path_factor)
    eigvec = f_dict['V'].to(device)
    path_ckpt = f_dict['ckpt']
    print('loading eigenvector of %s' % path_ckpt)

    g = torch.load(path_ckpt).to(device)
    '''state_dict = g.state_dict()
    g = Generator(512, 512, 8).to(device)
    g.load_state_dict(state_dict)'''

    trunc = g.mean_latent(4096)
    torch.save(trunc, 'trunc.pt')

    latent = torch.randn(1, 512, device=device)
    latent = g.get_latent(latent)
    torch.save(latent, 'latent.pt')
    print(latent[0, :10])

    indices = torch.arange(10)
    imgs = torch.zeros((3, len(indices) * imsize, (2 * m + 1) * imsize))
    for i in range(len(indices)):
        index = indices[i]
        direction = degree * eigvec[:, index].unsqueeze(0)

        scales = torch.linspace(-1, 1, 2 * m + 1)
        for j in range(len(scales)):
            scale = scales[j]
            # TODO 이거 이렇게 안 넣고 batch로 g에 넣어주면 되지 않나?
            img, _ = g([latent + scale * direction], truncation=truncation, truncation_latent=trunc,
                       input_is_='latent', randomize_noise=False)
            img = F.interpolate(img, (imsize, imsize))
            imgs[:, i * imsize:(i + 1) * imsize, j * imsize:(j + 1) * imsize] = img[0]

    path_save = './control/%s_degree-%f.png' % (out_prefix, degree)
    imgs = torch.clamp(imgs * 256, 0, 255).type(torch.uint8)
    imgs = TF.to_pil_image(imgs.permute((1, 2, 0)).numpy())
    imgs.save(path_save)
    print(path_save)


def main():
    torch.set_grad_enabled(False)
    args = parse_args()

    # eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    f_dict = torch.load(args.factor)
    eigvec = f_dict['V'].to(args.device)
    print('loading eigenvector of %s' % (f_dict['ckpt']))

    # ckpt = torch.load(args.ckpt)
    # g = Generator(args.size, 512, 8).to(args.device)
    # g.load_state_dict(ckpt["g_ema"], strict=False)
    g = torch.load(args.ckpt).to(args.device)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)

    imgs = []
    scales = torch.linspace(-1, 1, 2 * args.m + 1)
    for scale in scales:
        img, _ = g([latent + scale * direction], truncation=args.truncation, truncation_latent=trunc,
                   input_is='latent', )
        imgs.append(img)
    imgs = torch.cat(imgs, 0)

    path_save = './control/%s_index-%d_degree-%f.png' % (args.out_prefix, args.index, args.degree)
    grid = utils.save_image(imgs, path_save, normalize=True, range=(-1, 1), nrow=args.n_sample, )
    print(path_save)


if __name__ == "__main__":
    # main()
    compare_axes()
