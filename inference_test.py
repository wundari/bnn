# %%

import torch
import torch.utils.data as data
import torch._dynamo

# torch._dynamo.config.suppress_errors = True

import numpy as np
import random
import os
import matplotlib.pyplot as plt

from utilities.misc import NestedTensor
from config.config import BNNconfig

from modules.bnn import BNN

from dataset.scene_flow import SceneFlowFlyingThingsDataset
from dataset.scene_flow import SceneFlowMonkaaDataset

torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True

# %%
config = BNNconfig()
# get device
device = config.device

# fix the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# build model
model = BNN(config)
model = model.to(device)
# model = torch.compile(model) # disabled torch.compile during inference to avoid strange errors

# %% load pretrained model, if provided
pretrained_dir = os.path.join("run", config.dataset)
checkpoint = torch.load(
    f"{pretrained_dir}/epoch_4_model_{config.binocular_interaction}.pth.tar",
    map_location="cuda",
)
pretrained_dict = checkpoint["state_dict"]()

# fix the keys of the state dictionary
unwanted_prefix = "_orig_mod."
for k, v in list(pretrained_dict.items()):
    if k.startswith(unwanted_prefix):
        pretrained_dict[k[len(unwanted_prefix) :]] = pretrained_dict.pop(k)
model.load_state_dict(pretrained_dict)

# %%
parent_folder = "/media/wundari/S990Pro2_4TB"
datadir = f"{parent_folder}/Dataset/SceneFlow_complete/FlyingThings3D/"
dataset_train = SceneFlowFlyingThingsDataset(datadir, config, "train")
dataset_validation = SceneFlowFlyingThingsDataset(datadir, config, "validation")
# dataset_test = SceneFlowFlyingThingsDataset(datadir, config, "test")
# datadir = f"{parent_folder}/Dataset/SceneFlow_complete/Monkaa/"
# dataset_train = SceneFlowMonkaaDataset(datadir, config, "train")
# dataset_validation = SceneFlowMonkaaDataset(datadir, config, "validation")
# dataset_test = SceneFlowMonkaaDataset(datadir, config, "test")

data_loader_train = data.DataLoader(
    dataset_train,
    batch_size=config.batch_size,
    shuffle=True,
    # num_workers=2,
    pin_memory=True,
)

data_loader_validation = data.DataLoader(
    dataset_validation,
    batch_size=2,
    shuffle=True,
    # num_workers=2,
    pin_memory=True,
)
# data_loader_test = data.DataLoader(
#     dataset_test,
#     batch_size=2,
#     shuffle=True,
#     # num_workers=2,
#     pin_memory=True,
# )


# %% load image
inputs = next(iter(data_loader_validation))
# inputs = next(iter(data_loader_train))
left = inputs["left"]
right = inputs["right"]
disp = inputs["disp"]

# normalize to (0, 255), for visualization
img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

batch_size, c, h, w = left.size()
# visualize image
n_samples = 2
fig, axes = plt.subplots(nrows=n_samples, ncols=3)
for i in range(n_samples):
    # left image
    axes[i, 0].imshow(img_left[i].permute(1, 2, 0))
    axes[i, 0].axis("off")

    # right image
    axes[i, 1].imshow(img_right[i].permute(1, 2, 0))
    axes[i, 1].axis("off")

    # disparity ground truth
    vmin = -100
    vmax = 100
    temp = axes[i, 2].imshow(disp[i], cmap="jet", vmin=vmin, vmax=vmax)
    axes[i, 2].axis("off")
    # colorbar
    l_ax, b_ax, w_ax, h_ax = axes[i, 2].get_position().bounds
    cax = plt.gcf().add_axes([l_ax + w_ax + 0.03, b_ax, 0.03, h_ax])
    cbar_ticks = np.arange(vmin, vmax + 1, 50)
    cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticks)


# %% build nested tensor
# col_offset = config.downsample // 2
# row_offset = config.downsample // 2
# sampled_cols = (
#     torch.arange(col_offset, w, config.downsample)[None]
#     .expand(batch_size, -1)
#     .to(config.device)
# )
# sampled_rows = (
#     torch.arange(row_offset, h, config.downsample)[None]
#     .expand(batch_size, -1)
#     .to(config.device)
# )

input_data = NestedTensor(
    left.to(config.device),
    right.to(config.device),
    disp.to(config.device),
    sampled_cols=None,
    sampled_rows=None,
)
# %% inference
engine.model.eval()
with torch.autocast(device_type=config.device, dtype=torch.float16):
    disp_pred = engine.model(input_data)

# %% visualize output
# normalize to (0, 255), for visualization
img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

fig, axes = plt.subplots(nrows=batch_size, ncols=4, figsize=(12, 5))
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
    vmin = -50
    vmax = 50
    axes[i, 2].imshow(
        disp_pred[i].detach().cpu().numpy(), cmap="jet", vmin=vmin, vmax=vmax
    )
    axes[i, 2].set_title("Pred. disparity")
    axes[i, 2].axis("off")

    # disparity ground truth
    temp = axes[i, 3].imshow(disp[i], cmap="jet", vmin=vmin, vmax=vmax)
    axes[i, 3].set_title("Ground truth")
    axes[i, 3].axis("off")
    # colorbar
    l_ax, b_ax, w_ax, h_ax = axes[i, 3].get_position().bounds
    cax = plt.gcf().add_axes([l_ax + w_ax + 0.03, b_ax, 0.03, h_ax])
    cbar_ticks = np.arange(vmin, vmax + 1, 50)
    cbar = fig.colorbar(temp, cax=cax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_ticks)

plt.savefig(
    f"{engine.pred_images_dir}/output_val.pdf",
    dpi=600,
    bbox_inches="tight",
)
# plt.close()
# %%
