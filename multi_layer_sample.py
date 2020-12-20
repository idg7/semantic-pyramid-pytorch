import argparse
import itertools
import os

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset import Places365
from model import VGGFeature, Generator, Discriminator
from mask import make_mask, make_mask_pyramid
import distributed as dist


def requires_grad(module, flag):
    for m in module.parameters():
        m.requires_grad = flag


def d_ls_loss(real_predict, fake_predict):
    loss = (real_predict - 1).pow(2).mean() + fake_predict.pow(2).mean()

    return loss


def g_ls_loss(real_predict, fake_predict):
    loss = (fake_predict - 1).pow(2).mean()

    return loss


def recon_loss(features_fake, features_real, masks):
    r_loss = 0

    for f_fake, f_real, m in zip(features_fake, features_real, masks):
        if f_fake.ndim == 4:
            f_fake = F.max_pool2d(f_fake, 2, ceil_mode=True)
            f_real = F.max_pool2d(f_real, 2, ceil_mode=True)
            f_mask = F.max_pool2d(m, 2, ceil_mode=True)

            r_loss = (
                r_loss + (F.l1_loss(f_fake, f_real, reduction="none") * f_mask).mean()
            )

        else:
            r_loss = (
                r_loss
                + (F.l1_loss(f_fake, f_real, reduction="none") * m.squeeze(-1)).mean()
            )

    return r_loss


def diversity_loss(z1, z2, fake1, fake2, eps=1e-8):
    div_z = F.l1_loss(z1, z2, reduction="none").mean(1)
    div_fake = F.l1_loss(fake1, fake2, reduction="none").mean((1, 2, 3))

    d_loss = (div_z / (div_fake + eps)).mean()

    return d_loss


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)


def all_layers_combo(n_layers):
    combos = []
    for i in range(n_layers):
        combos.append(itertools.combinations(range(n_layers), i))
    return combos

def gen_masks(layer_combo, device):
    mask_batch = [];
    for i in layer_combo:
        masks = make_mask_pyramid(i, 7, ((112, 112), (56, 56), (28, 28), (14, 14), (7, 7)), device)
        mask_batch.append(masks)

    masks_zip = []
    for masks in zip(*mask_batch):
        masks_zip.append(torch.stack(masks, 0).unsqueeze(1))
    return masks_zip



def generate_multi_layers_sample(args, dataset, gen, dis, g_ema, device):

    vgg = VGGFeature("vgg16", [4, 9, 16, 23, 30], use_fc=True, checkpoint=args.checkpoint).eval().to(device)
    requires_grad(vgg, False)

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=4,
        sampler=dist.data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    loader_iter = sample_data(loader)

    pbar = range(args.start_iter, args.iter)

    if dist.get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True)

    with torch.no_grad():
        requires_grad(gen, False)
        requires_grad(dis, False)
        for i in pbar:
            real, class_id = next(loader_iter)

            real = real.to(device)
            class_id = class_id.to(device)
            for layer_combo in all_layers_combo(7):

                features, fcs = vgg(real)
                features = features + fcs[1:]
                masks = gen_masks(layer_combo, device)

                z1 = torch.randn(args.batch, args.dim_z, device=device)

                if args.distributed:
                    gen.broadcast_buffers = True

                fake1 = gen(z1, class_id, features, masks)

                if args.distributed:
                    gen.broadcast_buffers = False

                if dist.get_rank() == 0:
                    pbar.set_description(
                        f"i"
                    )
                utils.save_image(
                    fake1,
                    f"sample/{str(i).zfill(6)}-{layer_combo}.png",
                    nrow=int(args.batch ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )




if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='checkpoint/070000.pt')
    parser.add_argument("--iter", type=int, default=300)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dim_z", type=int, default=128)
    parser.add_argument("--dim_class", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, default='/home/administrator/experiments/high_importance_features_full_dataset/vgg16/models/120.pth')
    parser.add_argument("--path", type=str, default='/home/administrator/datasets/images_faces/faces_only_300/train')

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset = ImageFolder(args.path, transform=transform)
    args.n_class = 1000

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        dist.synchronize()

    gen = Generator(args.n_class, args.dim_z, args.dim_class).to(device)
    g_ema = Generator(args.n_class, args.dim_z, args.dim_class).to(device)
    accumulate(g_ema, gen, 0)
    dis = Discriminator(args.n_class).to(device)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        gen.load_state_dict(ckpt["g"])
        g_ema.load_state_dict(ckpt["g_ema"])
        dis.load_state_dict(ckpt["d"])

    if args.distributed:
        gen = nn.parallel.DistributedDataParallel(
            gen,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )

        dis = nn.parallel.DistributedDataParallel(
            dis,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=True,
        )

    generate_multi_layers_sample(args, dset, gen, dis, g_ema, device)
