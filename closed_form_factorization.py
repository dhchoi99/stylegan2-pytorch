import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="factor.pt", help='path to save')
    parser.add_argument("-- ckpt", type=str, help='path to ckpt')
    args = parser.parse_args()
    return args


def main2():
    args = parse_args()
    path_ckpt = 'checkpoint/twdne3.pt'

    g = torch.load(path_ckpt)
    s_dict = g.state_dict()

    modulate = {
        k: v
        for k, v in s_dict.items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    U, S, V = torch.svd(W)
    print('v', V.to('cpu').shape)

    torch.save({'ckpt':path_ckpt, 'U':U, 'S':S, 'V':V}, args.out)

def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt)
    checkpoints = ckpt.state_dict()
    modulate = {
        k: v
        for k, v in checkpoints.items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)


if __name__ == "__main__":
    # main()
    main2()
