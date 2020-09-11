import argparse
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from model import Generator


def main():
    # Set configs
    torch.set_grad_enabled(False)
    device = 'cuda'
    path_factor = 'factor.pt'
    m = 3
    degree = 10
    truncation = 0.7
    out_prefix = 'ganspace'
    imsize = 256

    f_dict = torch.load(path_factor)
    #path_ckpt = f_dict['ckpt']

    #path_ckpt = 'checkpoint/twdne3_dict.pt'
    #state_dict = torch.load(path_ckpt)
    path_ckpt = 'checkpoint/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pt'
    state_dict = torch.load(path_ckpt)['g']

    print('loading eigenvector of %s' % (path_ckpt))


    g = Generator(512, 512, 8).to(device)
    g.load_state_dict(state_dict)

    # TODO ganSpace의 code를 보았을 때 randn을 사용하는 것 같은데, randn과 rand의 차이 실험해보기
    num_rands = 1024
    zs = torch.rand(num_rands, 512).to(device)
    #zs = torch.randn(num_rands, 512).to(device)
    latents = g.style(zs)

    U, S, V = torch.pca_lowrank(latents, 512)
    V.to(device)

    trunc = g.mean_latent(4096)
    trunc = torch.load('trunc.pt')

    latent = torch.randn(1, 512, device=device)
    latent = g.get_latent(latent)
    latent = torch.load('latent.pt')
    print(latent[0, :10])

    indices = torch.arange(100)
    imgs = torch.zeros((3, len(indices) * imsize, (2 * m + 1) * imsize))
    for i in range(len(indices)):
        index = indices[i]
        direction = degree * V[:, index]

        scales = torch.linspace(-1, 1, 2 * m + 1)
        for j in range(len(scales)):
            scale = scales[j] * 1
            # TODO 이거 이렇게 안 넣고 batch로 g에 넣어주면 되지 않나?
            img, _ = g([latent + scale * direction], truncation=truncation, truncation_latent=trunc,
                       input_is_latent=True, randomize_noise=False)
            img = F.interpolate(img, (imsize, imsize))
            imgs[:, i * imsize:(i + 1) * imsize, j * imsize:(j + 1) * imsize] = img[0]

    path_save = './control/%s_degree-%f.png' % (out_prefix, degree)
    imgs = torch.clamp(imgs * 256, 0, 255).type(torch.uint8)
    imgs = TF.to_pil_image(imgs.permute((1, 2, 0)).numpy())
    imgs.save(path_save)
    print(path_save)


if __name__ == "__main__":
    main()
