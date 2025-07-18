# %%
import torch
import torch.nn as nn
import torch.utils.data as data

from config.config import BNNconfig

from modules.bnn import BNN

from dataset.scene_flow import SceneFlowFlyingThingsDataset
from dataset.scene_flow import SceneFlowMonkaaDataset

from utilities.misc import NestedTensor

import os
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

# %%
config = BNNconfig()
device = config.device

model = BNN(config)
model = model.to(device)
model = torch.compile(model)

# %% prepare dataset
parent_folder = "/media/wundari/S990Pro2_4TB"
# datadir = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/"
# dataset_train = SceneFlowFlyingThingsDataset(datadir, config, "train")
# dataset_validation = SceneFlowFlyingThingsDataset(datadir, config, "validation")
# dataset_test = SceneFlowFlyingThingsDataset(datadir, config, "test")
datadir = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/"
dataset_train = SceneFlowMonkaaDataset(datadir, config, "train")
dataset_validation = SceneFlowMonkaaDataset(datadir, config, "validation")
dataset_test = SceneFlowMonkaaDataset(datadir, config, "test")

data_loader_train = data.DataLoader(
    dataset_train,
    batch_size=config.batch_size,
    shuffle=True,
    # num_workers=2,
    pin_memory=True,
)
data_loader_validation = data.DataLoader(
    dataset_validation,
    batch_size=config.batch_size_val,
    shuffle=True,
    # num_workers=2,
    pin_memory=True,
)


# data_loader_test = data.DataLoader(
#     dataset_test,
#     batch_size=2,
#     shuffle=False,
#     # num_workers=2,
#     pin_memory=True,
# )
# %% set up optimizer
def configure_optimizers(model, weight_decay, learning_rate):

    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups.
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embedding decay,
    # all biases and layernorms dont't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
    )

    # create adamw optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    )

    return optimizer


def get_lr(it, config):
    # 1 linear warmup for warmup_iters steps
    if it < config.warmup_steps:
        return config.max_lr * (it + 1) / config.warmup_steps

    # 2 if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return config.min_lr

    # 3 in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_steps) / (max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


# optimizer = torch.optim.AdamW(
#     param_dicts, lr=config.lr, weight_decay=config.weight_decay
# )
optimizer = configure_optimizers(model, config.weight_decay, config.lr)
# %%
# sampled_cols = (
#     torch.arange(config.col_offset, config.w_crop, config.downsample)[None]
#     .expand(config.batch_size, -1)
#     .to(device)
# )
# sampled_rows = (
#     torch.arange(config.col_offset, config.h_crop, config.downsample)[None]
#     .expand(config.batch_size, -1)
#     .to(device)
# )

criterion = nn.SmoothL1Loss()
scaler = torch.cuda.amp.GradScaler()
max_steps = config.epochs * len(data_loader_train)
max_norm = 1.0
losses_train = []
losses_val = []

for epoch in range(config.epochs):
    tbar = tqdm(data_loader_train)
    for idx, inputs in enumerate(tbar):

        step = epoch * len(data_loader_train) + idx

        # once in a while evaluate validation loss
        if step % config.eval_interval == 0:
            model.eval()

            with torch.no_grad():
                inputs = next(iter(data_loader_validation))
                # bs, _, h, w = inputs["left"].size()

                inputs = NestedTensor(
                    inputs["left"].pin_memory().to(device, non_blocking=True),
                    inputs["right"].pin_memory().to(device, non_blocking=True),
                    disp=inputs["disp"].pin_memory().to(device, non_blocking=True),
                    ref=inputs["ref"].pin_memory().to(device, non_blocking=True),
                )

                with torch.autocast(device_type=device, dtype=torch.float16):
                    disp_pred = model(inputs)

                    # compute loss
                    loss_val = criterion(inputs.disp, disp_pred)

                # print(f'val L1 loss: {loss_val["l1"].item():.4f}')
                # gather validation L1 loss
                losses_val.append(loss_val.item())

        # training loop
        model.train()
        optimizer.zero_grad()

        inputs = next(iter(data_loader_train))
        # bs, _, h, w = inputs["left"].size()

        # sampled_cols = (
        #     torch.arange(config.col_offset, w, config.downsample)[None]
        #     .expand(bs, -1)
        #     .to(device)
        # )
        # sampled_rows = (
        #     torch.arange(config.row_offset, h, config.downsample)[None]
        #     .expand(bs, -1)
        #     .to(device)
        # )
        # build NestedTensor
        inputs = NestedTensor(
            inputs["left"].pin_memory().to(device, non_blocking=True),
            inputs["right"].pin_memory().to(device, non_blocking=True),
            disp=inputs["disp"].pin_memory().to(device, non_blocking=True),
            ref=inputs["ref"].pin_memory().to(device, non_blocking=True),
        )

        # forward pass
        with torch.autocast(device_type=device, dtype=torch.float16):
            disp_pred = model(inputs)

            # compute loss
            loss_train = criterion(inputs.disp, disp_pred)

        # backprop
        scaler.scale(loss_train).backward()

        # clip norm
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # step optimizer
        scaler.step(optimizer)

        # update the scale for next iteration
        scaler.update()

        # gather losses
        losses_train.append(loss_train.item())

        tbar.set_description(
            f"step {step}/{max_steps} |train loss: {loss_train.item():.4f} "
            + f"|val loss: {loss_val.item():.4f} |lr: {lr:.4e} "
        )
        # once in a while check predicted disp
        if step % config.eval_interval == 0:
            left = inputs.left
            right = inputs.right
            disp = inputs.disp
            # normalize to (0, 255), for visualization
            img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
            img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

            # visualize
            figsize = (16, 10)
            vmin = -100
            vmax = 100
            sns.set_theme()
            sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

            fig, axes = plt.subplots(nrows=4, ncols=4, figsize=figsize)
            for i in range(4):
                # left image
                axes[i, 0].imshow(img_left[i].cpu().permute(1, 2, 0))
                axes[i, 0].set_title("Left")

                # right image
                axes[i, 1].imshow(img_right[i].cpu().permute(1, 2, 0))
                axes[i, 1].set_title("Right")

                # predicted disparity
                axes[i, 2].imshow(
                    disp_pred[i].detach().cpu().numpy(),
                    cmap="jet",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[i, 2].set_title("Pred. disparity")

                # disparity ground truth
                temp = axes[i, 3].imshow(
                    disp[i].cpu(), cmap="jet", vmin=vmin, vmax=vmax
                )
                axes[i, 3].set_title("Ground truth")
                axes[i, 3].axis("off")
                # colorbar
                l_ax, b_ax, w_ax, h_ax = axes[i, 3].get_position().bounds
                cax = plt.gcf().add_axes([l_ax + w_ax + 0.03, b_ax, 0.03, h_ax])
                cbar_ticks = np.arange(vmin, vmax + 1, 50)
                cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
                cbar.ax.set_yticklabels(cbar_ticks)

            # turn off axis for all subplots
            for ax in axes.ravel():
                ax.set_axis_off()
            plt.show()

# %%save data
save_dir = os.path.join(
    "run",
    config.dataset,
    f"bino_interaction_{config.binocular_interaction}",
    "experiment_1",
)
checkpoint = {
    "epoch": epoch,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
filename = os.path.join(save_dir, f"epoch_{epoch}_model.pth.tar")
torch.save(checkpoint, filename)
np.save(f"{save_dir}/losses_train.npy", losses_train)
np.save(f"{save_dir}/losses_val.npy", losses_val)

# %% debug step by step

# feature extractor
from modules.stereo_encoder import build_feat_extractor

feature_extractor = build_feat_extractor(config)
feature_extractor = feature_extractor.to(device)

# feat = feature_extractor(inputs)
# extract features
feat_left = feature_extractor(inputs.left)
feat_right = feature_extractor(inputs.right)
feat_left.size()  # [4, 32, 128, 256]
# %% regressor
from modules.stereo_decoder import build_decoder

stereo_decoder = build_decoder(config)
stereo_decoder = stereo_decoder.to(device)

logits = stereo_decoder(feat_left, feat_right)
print(logits.size())  # [4, 1, 192, 256, 512]

# regress disparity
disp_pred = torch.sum(logits * model.disp_indices * inputs.ref.view(-1, 1, 1, 1), dim=1)
print(disp_pred.size())  # [4, 256, 512]
# %% debug cost volume
import torch.nn.functional as F

b, c, h, w = feat_left.size()

D = config.max_disp // 2
d = 20

padded_left = F.pad(feat_left, (D // 2, D // 2))
print(padded_left.size())

shifted_right = F.pad(feat_right, (d, D - d))
shifted_right.size()

plt.imshow(padded_left[0, 0].detach().cpu().numpy())
plt.imshow(feat_right[0, 0].detach().cpu().numpy())
plt.imshow(shifted_right[0, 0].detach().cpu().numpy())


# pad feat_left
padded_left = F.pad(feat_left, (D // 2, D // 2))

# Build cost volume
cost_volume = []

for d in range(D):

    padded_right = F.pad(feat_right, (d, D - d))

    if config.binocular_interaction == "mul":
        # bino-interaction: multiplication
        cost = padded_left * padded_right

    elif config.binocular_interaction == "sum_diff":
        # interaction: sum-diff channels
        cost = torch.cat(
            [padded_left + padded_right, padded_left - padded_right], dim=1
        )

    # crop to original size
    cost = cost[:, :, :, D // 2 : w + D // 2]

    # Aggregate cost
    cost = cost_aggregation(cost)  # [b, 1, h, w]
    cost_volume.append(cost.squeeze(1))  # [b, h, w]

# Stack cost volume
cost_volume = torch.stack(cost_volume, dim=1)  # [b, D, h, w]

# convTranspose2D to scale up the disparity channel
cost_volume = convT(cost_volume)  # [b, 2*D, h, w]
# cost_volume2 = convT(cost_volume)


# %%
base_channels = config.base_channels
cost_aggregation = nn.Sequential(
    nn.Conv2d(base_channels * 2 * 2, base_channels * 2, kernel_size=3, padding=1),
    nn.BatchNorm2d(base_channels * 2),
    nn.ReLU(inplace=True),
    nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(base_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
)
cost_aggregation = cost_aggregation.to(device)

convT = nn.ConvTranspose2d(
    config.max_disp // 2,
    config.max_disp // 2 * 2,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=1,
).to(device)

# %%
