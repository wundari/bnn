# %%
import torch
import torch._dynamo
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# torch._dynamo.config.suppress_errors = True

import numpy as np
import random
import os
import matplotlib.pyplot as plt

from module.sttr import STTR
from utilities.misc import NestedTensor
from config.config import BNNconfig

from RDS.DataHandler_RDS import RDS_Handler, DatasetRDS

torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True


# %%
config = BNNconfig()

# get device
device = "cuda"

# fix the seed for reproducibility
seed = 42
torch.manual_seed(seed) 
np.random.seed(seed)
random.seed(seed)

# build model
model = STTR(config)
model = model.to(device)
# model = torch.compile(model) # disabled torch.compile during inference to avoid strange errors

# %% load pretrained model, if provided
pretrained_dir = os.path.join("run", config.dataset, config.checkpoint, "experiment_20")
checkpoint = torch.load(f"{pretrained_dir}/epoch_2_model.pth.tar")
pretrained_dict = checkpoint["state_dict"]

# fix the keys of the state dictionary
unwanted_prefix = "_orig_mod."
for k, v in list(pretrained_dict.items()):
    if k.startswith(unwanted_prefix):
        pretrained_dict[k[len(unwanted_prefix) :]] = pretrained_dict.pop(k)
model.load_state_dict(pretrained_dict)

# %% RDS dataloader
dotMatch = 0.0
dotDens = 0.25
target_disp = 10  # RDS target disparity (pix) to be analyzed
disp_ct_pix_list = [target_disp, -target_disp]
n_rds_each_disp = 64  # n_rds for each disparity magnitude in disp_ct_pix
background_flag = 1  # 1: with cRDS background
pedestal_flag = 0  # 1: use pedestal to ensure rds disparity > 0

rds_left, rds_right, rds_label = RDS_Handler.generate_rds(
    dotMatch,
    dotDens,
    disp_ct_pix_list,
    n_rds_each_disp,
    background_flag,
    pedestal_flag,
)

transform_data = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda t: (t + 1.0) / 2.0)]
)

rds_data = DatasetRDS(rds_left, rds_right, rds_label, transform=transform_data)
rds_loader = DataLoader(
    rds_data,
    batch_size=2,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=1,
)


# %% load image
left, right, disp = next(iter(rds_loader))
print(disp)

# normalize to (0, 255), for visualization
img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

batch_size, c, h, w = left.size()
# visualize image
fig, axes = plt.subplots(nrows=batch_size, ncols=2)
for i in range(batch_size):
    # left image
    axes[i, 0].imshow(img_left[i].permute(1, 2, 0))
    axes[i, 0].axis("off")

    # right image
    axes[i, 1].imshow(img_right[i].permute(1, 2, 0))
    axes[i, 1].axis("off")

    # # disparity ground truth
    # vmin = 0
    # vmax = 200
    # temp = axes[i, 2].imshow(disp[i], cmap="jet", vmin=vmin, vmax=vmax)
    # axes[i, 2].axis("off")
    # # colorbar
    # l_ax, b_ax, w_ax, h_ax = axes[i, 2].get_position().bounds
    # cax = plt.gcf().add_axes([l_ax + w_ax + 0.03, b_ax, 0.03, h_ax])
    # cbar_ticks = np.arange(vmin, vmax + 1, 50)
    # cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
    # cbar.ax.set_yticklabels(cbar_ticks)


# %% build nested tensor
col_offset = config.downsample // 2
row_offset = config.downsample // 2
sampled_cols = (
    torch.arange(col_offset, w, config.downsample)[None]
    .expand(batch_size, -1)
    .to(config.device)
)
sampled_rows = (
    torch.arange(row_offset, h, config.downsample)[None]
    .expand(batch_size, -1)
    .to(config.device)
)

input_data = NestedTensor(
    left.to(config.device),
    right.to(config.device),
    sampled_cols=None,
    sampled_rows=None,
)
# %% inference
with torch.autocast(device_type=config.device, dtype=torch.float16):
    output = model(input_data)

# %% visualize output
# normalize to (0, 255), for visualization
img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(10, 5))
fig.text(0.5, 0.9, f"RDS dot match: {dotMatch}, dot dens: {dotDens}")
for i in range(batch_size):
    # left image
    axes[i, 0].imshow(img_left[i].permute(1, 2, 0))
    axes[i, 0].set_title("Left")
    axes[i, 0].axis("off")

    # right image
    axes[i, 1].imshow(img_right[i].permute(1, 2, 0))
    axes[i, 1].set_title("Right")
    axes[i, 1].axis("off")

    # predicted disparity
    vmin = -10
    vmax = 10
    # set disparity of occluded area to 0
    disp_pred = output.data.cpu().numpy()[i]
    
    temp = axes[i, 2].imshow(disp_pred, cmap="jet", vmin=vmin, vmax=vmax)
    axes[i, 2].set_title(f"Pred. disparity (target: {disp[i]} pix)")
    axes[i, 2].axis("off")

    # # disparity ground truth
    # temp = axes[i, 3].imshow(disp[i], cmap="jet", vmin=vmin, vmax=vmax)
    # axes[i, 3].set_title("Ground truth")
    # axes[i, 3].axis("off")
    # colorbar
    l_ax, b_ax, w_ax, h_ax = axes[i, 2].get_position().bounds
    cax = plt.gcf().add_axes([l_ax + w_ax + 0.03, b_ax, 0.03, h_ax])
    cbar_ticks = np.arange(vmin, vmax + 1, 5)
    cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticks)


# %% compute metrics
# manually compute occluded region
coord = np.linspace(0, w - 1, w)[None,]  # 1xW
shifted_coord = coord - disp.numpy()
occ_mask = shifted_coord < 0  # occlusion mask, 1 indicates occ

plt.imshow(occ_mask[0])

# %% compute difference in non-occluded region only
diff = disp - disp_pred
diff[0][occ_mask[0]] = 0.0
diff[1][occ_mask[1]] = 0.0

i = 1
valid_mask = np.logical_and(disp[i] > 0.0, ~occ_mask[i])

# calculate 3-pix error
pix_err = (diff[i] > 3).sum()
pix_total = valid_mask.sum()
print(f"3-pix error: {pix_err / pix_total * 100:.2f}%")

# calculate epe
err = np.abs(diff[i][valid_mask]).sum()
print(f"EPE: {err / pix_total: .2f}")

# %%
