# %% import necessary modules

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch._dynamo

import numpy as np
import random
import math
import sys
import inspect

from module.sttr import STTR
from module.loss import build_criterion

from utilities.foward_pass import forward_pass, set_downsample, write_summary
from utilities.inference import inference
from utilities.eval import evaluate
from utilities.foward_pass import write_summary
from utilities.summary_logger import TensorboardSummary
from utilities.checkpoint_saver_repro import Saver
from utilities.misc import NestedTensor

from config.config import STTRconfig

from dataset.scene_flow import SceneFlowFlyingThingsDataset
from dataset.scene_flow import SceneFlowMonkaaDataset

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True


# %%
def print_param(model):
    """
    print number of parameters in the model
    """

    n_parameters = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "backbone" in n and p.requires_grad
    )
    print("number of params in backbone:", f"{n_parameters:,}")
    n_parameters = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "transformer" in n and "regression" not in n and p.requires_grad
    )
    print("number of params in transformer:", f"{n_parameters:,}")
    n_parameters = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "tokenizer" in n and p.requires_grad
    )

    print("number of params in tokenizer:", f"{n_parameters:,}")
    n_parameters = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "regression" in n and p.requires_grad
    )
    print("number of params in regression:", f"{n_parameters:,}")


# %%
config = STTRconfig()

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
model = torch.compile(model)
print_param(model)

# %%
# set learning rate
param_dicts = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if "backbone" not in n and "regression" not in n and p.requires_grad
        ]
    },
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if "backbone" in n and p.requires_grad
        ],
        "lr": config.lr_backbone,
    },
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if "regression" in n and p.requires_grad
        ],
        "lr": config.lr_regression,
    },
]


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


# optimizer = torch.optim.AdamW(
#     param_dicts, lr=config.lr, weight_decay=config.weight_decay
# )
optimizer = configure_optimizers(model, config.weight_decay, config.lr)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=config.lr_decay_rate
)

# %%
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
    batch_size=2,
    shuffle=False,
    # num_workers=2,
    pin_memory=True,
)
data_loader_test = data.DataLoader(
    dataset_test,
    batch_size=2,
    shuffle=False,
    # num_workers=2,
    pin_memory=True,
)

# %% check images
import os
from utilities.python_pfm import readPFM
from PIL import Image
from natsort import natsorted
import matplotlib.pyplot as plt

split_folder = "TRAIN"
directory = os.path.join(datadir, "frames_cleanpass", split_folder)
sub_folders = [
    os.path.join(directory, subset)
    for subset in os.listdir(directory)
    if os.path.isdir(os.path.join(directory, subset))
]

seq_folders = []
for sub_folder in sub_folders:
    seq_folders += [
        os.path.join(sub_folder, seq)
        for seq in os.listdir(sub_folder)
        if os.path.isdir(os.path.join(sub_folder, seq))
    ]

left_data = []
for seq_folder in seq_folders:
    left_data += [
        os.path.join(seq_folder, "left", img)
        for img in os.listdir(os.path.join(seq_folder, "left"))
    ]
left_data = natsorted(left_data)

directory = os.path.join(datadir, "occlusion", split_folder, "left")
occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
occ_data = natsorted(occ_data)

n_img = 10
c = 10000
fig, axes = plt.subplots(nrows=n_img, ncols=4, figsize=(5, 10))
for i in range(c, c + n_img):
    img_left_name = left_data[i]
    img_right_name = img_left_name.replace("left", "right")
    disp_name = img_left_name.replace("frames_cleanpass", "disparity").replace(
        "png", "pfm"
    )
    occ_name = occ_data[i]

    img_left = Image.open(img_left_name)
    img_right = Image.open(img_right_name)
    img_disp, _ = readPFM(disp_name)
    img_occ = Image.open(occ_name)

    axes[i - c, 0].imshow(img_left)
    axes[i - c, 0].axis("off")
    axes[i - c, 1].imshow(img_right)
    axes[i - c, 1].axis("off")
    axes[i - c, 2].imshow(img_disp)
    axes[i - c, 2].axis("off")
    axes[i - c, 3].imshow(img_occ)
    axes[i - c, 3].axis("off")

# %% load pretrained model, if provided
# pretrained_dir = os.path.join("run", config.dataset, config.checkpoint, "experiment_4")
# if config.resume != "":

#     checkpoint = torch.load(f"{pretrained_dir}/epoch_4_model.pth.tar")
#     pretrained_dict = checkpoint["state_dict"]

#     # fix the keys of the state dictionary
#     unwanted_prefix = "_orig_mod."
#     for k, v in list(pretrained_dict.items()):
#         if k.startswith(unwanted_prefix):
#             pretrained_dict[k[len(unwanted_prefix) :]] = pretrained_dict.pop(k)
#     model.load_state_dict(pretrained_dict)

# %% set downsample rate
set_downsample(config)

# build loss criterion
criterion = build_criterion(config)

# %% inference
# inference(model, data_loader_test, device, config.downsample)

# %% initiate saver and logger
checkpoint_saver = Saver(config)
summary = TensorboardSummary(checkpoint_saver.experiment_dir)

# %% eval
# evaluate(model, criterion, data_loader_validation, device, 0, summary, True)


# %% train
def save_checkpoint(epoch, model, optimizer, lr_scheduler, checkpoint_saver, best):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        # "best_pred": prev_best,
    }

    if best:
        checkpoint_saver.save_checkpoint(checkpoint, "model.pth.tar", write_best=False)
    else:
        checkpoint_saver.save_checkpoint(
            checkpoint, "epoch_" + str(epoch) + "_model.pth.tar", write_best=False
        )


# learning rate setup
max_steps = config.epochs * len(data_loader_train)


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


train_stats = {
    "l1": 0.0,
    "occ_be": 0.0,
    "l1_raw": 0.0,
    "iou": 0.0,
    "rr": 0.0,
    "epe": 0.0,
    "error_px": 0.0,
    "total_px": 0.0,
}

scaler = torch.cuda.amp.GradScaler()
max_norm = 1.0
losses_train = []
losses_val = []
for epoch in range(config.start_epoch, config.epochs):
    tbar = tqdm(data_loader_train)
    for idx, inputs in enumerate(tbar):

        step = epoch * len(data_loader_train) + idx

        # once in a while evaluate validation loss
        if step % config.eval_interval == 0:
            model.eval()

            with torch.no_grad():
                inputs = next(iter(data_loader_validation))
                bs, _, h, w = inputs["left"].size()

                sampled_cols = (
                    torch.arange(config.col_offset, w, config.downsample)[None]
                    .expand(bs, -1)
                    .to(device)
                )
                sampled_rows = (
                    torch.arange(config.row_offset, h, config.downsample)[None]
                    .expand(bs, -1)
                    .to(device)
                )

                # build NestedTensor
                # inputs = NestedTensor(
                #     inputs["left"].to(device),
                #     inputs["right"].to(device),
                #     sampled_cols=sampled_cols,
                #     sampled_rows=sampled_rows,
                #     disp=inputs["disp"].to(device),
                #     occ_mask=inputs["occ_mask"].to(device),
                #     occ_mask_right=inputs["occ_mask_right"].to(device),
                # )
                inputs = NestedTensor(
                    inputs["left"].to(device),
                    inputs["right"].to(device),
                    sampled_cols=sampled_cols,
                    sampled_rows=sampled_rows,
                    disp=inputs["disp"].to(device),
                    occ_mask=None,
                    occ_mask_right=None,
                )

                with torch.autocast(device_type=device, dtype=torch.float16):
                    outputs = model(inputs)

                    # compute loss
                    loss_val = criterion(inputs, outputs)

                # print(f'val L1 loss: {loss_val["l1"].item():.4f}')
                # gather validation L1 loss
                losses_val.append(loss_val["aggregated"].item())

        # training loop
        model.train()
        optimizer.zero_grad()

        inputs = next(iter(data_loader_train))
        # print(inputs["left"].size())

        # forward pass
        outputs, loss_train, sampled_disp = forward_pass(
            model, inputs, device, criterion, train_stats
        )

        # gather training L1 loss each iteration
        losses_train.append(loss_train["aggregated"].item())

        # terminate training if exploded
        if not math.isfinite(loss_train["aggregated"].item()):
            print(f'Loss is {loss_train["aggregated"].item()}, stopping training')
            sys.exit(1)

        # backprop
        scaler.scale(loss_train["aggregated"]).backward()

        # clip norm
        # if max_norm > 0:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # step optimizer
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
        torch.cuda.synchronize()

        tbar.set_description(
            f'step {step}/{max_steps} |train loss: {loss_train["aggregated"].item():.4f} '
            + f'|val loss: {loss_val["aggregated"].item():.4f} |lr: {lr:.4e} '
            + f'|pixel_error:{loss_train["error_px"] / loss_train["total_px"]:.4f}'
        )

        # print(
        # f'step {step}/{max_steps}, |train loss: {loss_train["aggregated"].item():.4f}, '
        # + f'|val loss: {loss_val["aggregated"].item():.4f}, |lr: {lr:.4e}, '
        # + f'|pixel_error:{loss_train["error_px"] / loss_train["total_px"]:.4f}'
        # )

        # clear cache
        torch.cuda.empty_cache()

        ## once a while check predicted disparity
        if step % config.eval_interval == 0:
            left = inputs["left"]
            right = inputs["right"]
            disp = inputs["disp"]
            # normalize to (0, 255), for visualization
            img_left = ((left / left.max()) * 128 + 127).to(torch.uint8)
            img_right = ((right / right.max()) * 128 + 127).to(torch.uint8)

            # visualize
            figsize = (16, 10)
            vmin = -100
            vmax = 100
            sns.set_theme()
            sns.set_theme(context="paper", style="white", font_scale=2, palette="deep")

            fig, axes = plt.subplots(nrows=config.batch_size, ncols=4, figsize=figsize)
            for i in range(config.batch_size):
                # left image
                axes[i, 0].imshow(img_left[i].permute(1, 2, 0))
                axes[i, 0].set_title("Left")
                # axes[i, 0].axis("off")

                # right image
                axes[i, 1].imshow(img_right[i].permute(1, 2, 0))
                axes[i, 1].set_title("Right")
                # axes[i, 1].axis("off")

                # predicted disparity
                # set disparity of occluded area to 0
                disp_pred = outputs["disp_pred"].data.cpu().numpy()[i]
                # occ_pred = output["occ_pred"].data.cpu().numpy()[i] > 0.5
                # disp_pred[occ_pred] = 0.0
                axes[i, 2].imshow(disp_pred, cmap="jet", vmin=vmin, vmax=vmax)
                axes[i, 2].set_title("Pred. disparity")
                # axes[i, 2].axis("off")

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

            # turn off axis for all subplots
            for ax in axes.ravel():
                ax.set_axis_off()

            if not os.path.exists(f"{checkpoint_saver.experiment_dir}/pred_images"):
                os.makedirs(f"{checkpoint_saver.experiment_dir}/pred_images")

            plt.savefig(
                f"{checkpoint_saver.experiment_dir}/pred_images/output.pdf",
                bbox_inches="tight",
            )

            plt.close()

    # compute avg
    train_stats["px_error_rate"] = train_stats["error_px"] / train_stats["total_px"]

    # log to tensorboard
    write_summary(train_stats, summary, epoch, "train")

    print(
        "Training loss",
        train_stats["l1"],
        "pixel error rate",
        train_stats["px_error_rate"],
    )
    print("RR loss", train_stats["rr"])

    # save model each epoch
    save_checkpoint(epoch, model, optimizer, lr_scheduler, checkpoint_saver, False)
# save train and val losses
np.save(f"{checkpoint_saver.experiment_dir}/losses_train.npy", losses_train)
np.save(f"{checkpoint_saver.experiment_dir}/losses_val.npy", losses_val)

# %% debug step by step

# # prepare data
# inputs = next(iter(data_loader_train))
# left = inputs["left"].to(device)
# right = inputs["right"].to(device)
# disp, occ_mask, occ_mask_right = (
#     inputs["disp"].to(device),
#     inputs["occ_mask"].to(device),
#     inputs["occ_mask_right"].to(device),
# )

# # if need to downsample, sample with a provided stride
# bs, _, h, w = left.size()
# if config.downsample <= 0:
#     sampled_cols = None
#     sampled_rows = None
# else:
#     col_offset = config.downsample // 2
#     row_offset = config.downsample // 2
#     sampled_cols = (
#         torch.arange(col_offset, w, config.downsample)[None].expand(bs, -1).to(device)
#     )
#     sampled_rows = (
#         torch.arange(row_offset, h, config.downsample)[None].expand(bs, -1).to(device)
#     )

# # build the input
# inputs = NestedTensor(
#     left,  # .to("cpu"),
#     right,  # .to("cpu"),
#     sampled_cols=sampled_cols,  # .to("cpu"),
#     sampled_rows=sampled_rows,  # .to("cpu"),
#     disp=disp,
#     occ_mask=occ_mask,
#     occ_mask_right=occ_mask_right,
# )

# # %% feature extractor
# feat = model.backbone(inputs)
# # list 2BxCxHxW = [2xbatch_size, RGB_channel, 360, 640]
# # src_stereo = torch.cat([inputs.left, inputs.right], dim=0)

# # # in conv
# # output = model.backbone.in_conv(src_stereo)  # [2B, 32, 183, 390]

# # # res blocks
# # output_1 = model.backbone.resblock_1(output)  # 1/4, [2B, 64, 92, 195]
# # output_2 = model.backbone.resblock_2(output_1)  # 1/8, [2B, 128, 46, 98]

# # # spp
# # import math
# # import torch.nn.functional as F

# # h_spp, w_spp = math.ceil(h / 16), math.ceil(w / 16)
# # spp_1 = model.backbone.branch1(output_2)  # [2B, 32, 2, 6]
# # spp_1 = F.interpolate(
# #     spp_1, size=(h_spp, w_spp), mode="bilinear", align_corners=False
# # )  # [2B, 32, 23, 49]
# # spp_2 = model.backbone.branch2(output_2)  # [2B, 32, 5, 12]
# # spp_2 = F.interpolate(
# #     spp_2, size=(h_spp, w_spp), mode="bilinear", align_corners=False
# # )  # [2B, 32, 23, 49]
# # spp_3 = model.backbone.branch3(output_2)  # [2B, 32, 11, 24]
# # spp_3 = F.interpolate(
# #     spp_3, size=(h_spp, w_spp), mode="bilinear", align_corners=False
# # )  # [2B, 32, 23, 49]
# # spp_4 = model.backbone.branch4(output_2)  # [2B, 32, 23, 49]
# # spp_4 = F.interpolate(
# #     spp_4, size=(h_spp, w_spp), mode="bilinear", align_corners=False
# # )  # [2B, 32, 23, 49]
# # output_3 = torch.cat([spp_1, spp_2, spp_3, spp_4], dim=1)  # [2B, 128, 23, 49]

# # feature backbone output
# # feat = [src_stereo, output_1, output_2, output_3]

# # %% tokenizer
# tokens = model.tokenizer(feat)  # [2B, 128, 360, 640]
# # tokens = torch.randn((2*config.batch_size, 128, 360, 640), dtype=torch.float32)

# # growth_rate = 4
# # block_config = [4, 4, 4, 4]
# # layer_channel = [64, 128, 128]
# # num_resolution = len(layer_channel)
# # hidden_dim = config.channel_dim

# # layer_channel.reverse()
# # block_config.reverse()


# # bottle_neck = _DenseBlock(
# #     block_config[0],
# #     layer_channel[0],
# #     4,
# #     drop_rate=0.0,
# #     growth_rate=growth_rate,
# # )

# # up = []
# # dense_block = []
# # prev_block_channels = growth_rate * block_config[0]
# # for i in range(num_resolution):
# #     if i == num_resolution - 1:
# #         up.append(TransitionUp(prev_block_channels, hidden_dim, 4))
# #         dense_block.append(DoubleConv(hidden_dim + 3, hidden_dim))
# #     else:
# #         up.append(TransitionUp(prev_block_channels, prev_block_channels))
# #         cur_channels_count = prev_block_channels + layer_channel[i + 1]
# #         dense_block.append(
# #             _DenseBlock(
# #                 block_config[i + 1],
# #                 cur_channels_count,
# #                 4,
# #                 drop_rate=0.0,
# #                 growth_rate=growth_rate,
# #             )
# #         )
# #         prev_block_channels = growth_rate * block_config[i + 1]

# # up = nn.ModuleList(up)
# # dense_block = nn.ModuleList(dense_block)

# # feat.reverse()
# # output = model.tokenizer.bottle_neck(feat[0])  # [2B, 144, 23, 49]
# # output = output[:, -(block_config[0] * growth_rate) :]  # [2B, 16, 23, 49]

# # for i in range(num_resolution):
# #     hs = up[i](output, feat[i + 1])  # scale up and concat
# #     output = dense_block[i](hs)  # denseblock

# #     if i < num_resolution - 1:  # other than the last convolution block
# #         output = output[
# #             :, -(block_config[i + 1] * growth_rate) :
# #         ]  # take only the new features

# # %% transformer
# from utilities.misc import batched_index_select

# # position encoding
# pos_enc = model.pos_encoder(inputs)  # [425, 128] -> size after downsampling
# # pos_enc = torch.randn((425, 128), dtype=torch.float32).to(device)

# # separate left and right
# feat_left = tokens[:bs]  # .to(device)
# feat_right = tokens[bs:]  # .to(device)  # [B=4, 128, 360, 640]

# # downsample
# if inputs.sampled_cols is not None:
#     feat_left = batched_index_select(
#         feat_left, 3, inputs.sampled_cols
#     )  # [B=4, 128, 120, 213]
#     feat_right = batched_index_select(feat_right, 3, inputs.sampled_cols)
# if inputs.sampled_rows is not None:
#     feat_left = batched_index_select(feat_left, 2, inputs.sampled_rows)
#     feat_right = batched_index_select(feat_right, 2, inputs.sampled_rows)

# # transformer
# # attn_weight = model.transformer(
# #     feat_left.to("cpu"), feat_right.to("cpu"), pos_enc.to("cpu")
# # )
# attn_weight = model.transformer(feat_left, feat_right, pos_enc)

# # debug transformer

# # flatten NxCxHxW to WxHNxC
# bs, c, hn, w = feat_left.shape
# # w = 640

# feat_left = (
#     feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
# )  # [2B, C, H, W] => [C, W, H, 2B] => [C, W, Hx2B] => [W, Hx2B, C]: [213, 480, 128]
# feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
# if pos_enc is not None:
#     with torch.no_grad():
#         # indexes to shift rel pos encoding
#         indexes_r = torch.linspace(w - 1, 0, w).view(w, 1)  # .to(feat_left.device)
#         indexes_c = torch.linspace(0, w - 1, w).view(1, w)  # .to(feat_left.device)
#         pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
#         # pos_indexes = pos_indexes.to(device)
# else:
#     pos_indexes = None

# # concatenate left and right features
# feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2BNxC, [213, 960, 128]
# # feat = torch.randn((213, 960, 128), dtype=torch.float32).to(device)


# # %% debug self-attention
# hidden_dim = 128
# nhead = 8
# self_attn_layers = TransformerSelfAttnLayer(hidden_dim, nhead)
# # self_attn_layers = self_attn_layers.to(device)
# feat2 = self_attn_layers.norm1(feat)
# feat2, attn_weight, _ = self_attn_layers.self_attn(
#     query=feat2,
#     key=feat2,
#     value=feat2,
#     pos_enc=pos_enc,
#     pos_indexes=pos_indexes,
# )

# query = feat2
# key = feat2
# value = feat2

# # alternating
# for idx, (self_attn, cross_attn) in enumerate(
#     zip(model.transformer.self_attn_layers, model.transformer.cross_attn_layers)
# ):
#     print(f"{idx}: {self_attn}")

#     # self-attention
#     feat = self_attn(
#         feat.to("cpu"), pos_enc.to("cpu"), pos_indexes.to("cpu")
#     )  # [213, 960, 128]

#     # cross-attention
#     if idx == model.transformer.num_attn_layers - 1:
#         last_layer = True
#     else:
#         last_layer = False
#     feat, raw_attn = cross_attn(
#         feat[:, :480].to("cpu"),
#         feat[:, 480:].to("cpu"),
#         pos_enc.to("cpu"),
#         pos_indexes.to("cpu"),
#         last_layer,
#     )

# attn_weight = model.transformer._alternating_attn(
#     feat.to("cpu"), pos_enc.to("cpu"), pos_indexes.to("cpu"), 480
# )
# attn_weight = attn_weight.view(hn, bs, w, w).permute(1, 0, 2, 3)

# # %%
# w, bsz, embed_dim = feat2.size()
# head_dim = embed_dim // nhead
# assert head_dim * nhead == embed_dim, "embed_dim must be divisible by num_heads"

# # project to get qkv
# if torch.equal(query, key) and torch.equal(key, value):
#     # self-attention
#     q, k, v = F.linear(
#         query,
#         self_attn_layers.self_attn.in_proj_weight,
#         self_attn_layers.self_attn.in_proj_bias,
#     ).chunk(3, dim=-1)

# # project to find q_r, k_r
# if pos_enc is not None:
#     # reshape pos_enc
#     pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(
#         w, w, -1
#     )  # 2W-1xC -> WW'xC -> WxW'xC
#     # compute k_r, q_r
#     _start = 0
#     _end = 2 * embed_dim
#     _w = self_attn_layers.self_attn.in_proj_weight[_start:_end, :]
#     _b = self_attn_layers.self_attn.in_proj_bias[_start:_end]
#     q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # WxW'xC
# else:
#     q_r = None
#     k_r = None

# # scale query
# scaling = float(head_dim) ** -0.5
# q = q * scaling
# if q_r is not None:
#     q_r = q_r * scaling

# # reshape
# q = q.contiguous().view(
#     w, bsz, self_attn_layers.self_attn.num_heads, head_dim
# )  # WxNxExC

# if k is not None:
#     k = k.contiguous().view(-1, bsz, self_attn_layers.self_attn.num_heads, head_dim)
# if v is not None:
#     v = v.contiguous().view(-1, bsz, self_attn_layers.self_attn.num_heads, head_dim)

# if q_r is not None:
#     q_r = q_r.contiguous().view(
#         w, w, self_attn_layers.self_attn.num_heads, head_dim
#     )  # WxW'xExC
# if k_r is not None:
#     k_r = k_r.contiguous().view(w, w, self_attn_layers.self_attn.num_heads, head_dim)

# # compute attn weight
# attn_feat = torch.einsum("wnec,vnec->newv", q, k)  # NxExWxW'

# # add positional terms
# if pos_enc is not None:
#     # 0.3 s
#     attn_feat_pos = torch.einsum("wnec,wvec->newv", q, k_r)  # NxExWxW'
#     attn_pos_feat = torch.einsum("vnec,wvec->newv", k, q_r)  # NxExWxW'

#     # 0.1 s
#     attn = attn_feat + attn_feat_pos + attn_pos_feat
# else:
#     attn = attn_feat

# assert list(attn.size()) == [bsz, self.num_heads, w, w]

# # apply attn mask
# if attn_mask is not None:
#     attn_mask = attn_mask[None, None, ...]
#     attn += attn_mask

# # raw attn
# raw_attn = attn

# # softmax
# attn = F.softmax(attn, dim=-1)

# # compute v, equivalent to einsum('',attn,v),
# # need to do this because apex does not support einsum when precision is mixed
# v_o = torch.bmm(
#     attn.view(bsz * self.num_heads, w, w),
#     v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w, head_dim),
# )  # NxExWxW', W'xNxExC -> NExWxC
# assert list(v_o.size()) == [bsz * self.num_heads, w, head_dim]
# v_o = (
#     v_o.reshape(bsz, self.num_heads, w, head_dim)
#     .permute(2, 0, 1, 3)
#     .reshape(w, bsz, embed_dim)
# )
# v_o = F.linear(v_o, self.out_proj.weight, self.out_proj.bias)

# # average attention weights over heads
# attn = attn.sum(dim=1) / self.num_heads

# # raw attn
# raw_attn = raw_attn.sum(dim=1)


# # %% debug cross-attention
# hidden_dim = 128
# nhead = 8
# cross_attn_layers = TransformerCrossAttnLayer(hidden_dim, nhead)
# # cross_attn_layers = cross_attn_layers.to(device)
# feat_left = torch.randn((213, 480, 128), dtype=torch.float32)
# feat_right = torch.randn((213, 480, 128), dtype=torch.float32)

# feat_left_2 = cross_attn_layers.norm1(feat_left)
# feat_right_2 = cross_attn_layers.norm1(feat_right)

# feat2, attn_weight, _ = cross_attn_layers.cross_attn(
#     query=feat_right_2,
#     key=feat_left_2,
#     value=feat_left_2,
#     pos_enc=pos_enc,
#     pos_indexes=pos_indexes.to("cpu"),
# )

# query = feat_right_2
# key = feat_left_2
# value = feat_left_2
# # start cross-attention
# w, bsz, embed_dim = query.size()
# head_dim = embed_dim // nhead
# assert head_dim * nhead == embed_dim, "embed_dim must be divisible by num_heads"

# # project to get qkv

# if torch.equal(key, value):
#     # cross-attention
#     _b = cross_attn_layers.cross_attn.in_proj_bias
#     _start = 0
#     _end = embed_dim
#     _w = cross_attn_layers.cross_attn.in_proj_weight[_start:_end, :]
#     if _b is not None:
#         _b = _b[_start:_end]
#     q = F.linear(query, _w, _b)

#     if key is None:
#         assert value is None
#         k = None
#         v = None
#     else:
#         _b = cross_attn_layers.cross_attn.in_proj_bias
#         _start = embed_dim
#         _end = None
#         _w = cross_attn_layers.cross_attn.in_proj_weight[_start:, :]
#         if _b is not None:
#             _b = _b[_start:]
#         k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

# # project to find q_r, k_r
# if pos_enc is not None:
#     # reshape pos_enc
#     pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(
#         w, w, -1
#     )  # 2W-1xC -> WW'xC -> WxW'xC
#     # compute k_r, q_r
#     _start = 0
#     _end = 2 * embed_dim
#     _w = cross_attn_layers.cross_attn.in_proj_weight[_start:_end, :]
#     _b = cross_attn_layers.cross_attn.in_proj_bias[_start:_end]
#     q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # WxW'xC
# else:
#     q_r = None
#     k_r = None

# # scale query
# scaling = float(head_dim) ** -0.5
# q = q * scaling
# if q_r is not None:
#     q_r = q_r * scaling

# # reshape
# q = q.contiguous().view(
#     w, bsz, cross_attn_layers.cross_attn.num_heads, head_dim
# )  # WxNxExC

# if k is not None:
#     k = k.contiguous().view(-1, bsz, cross_attn_layers.cross_attn.num_heads, head_dim)
# if v is not None:
#     v = v.contiguous().view(-1, bsz, cross_attn_layers.cross_attn.num_heads, head_dim)

# if q_r is not None:
#     q_r = q_r.contiguous().view(
#         w, w, cross_attn_layers.cross_attn.num_heads, head_dim
#     )  # WxW'xExC
# if k_r is not None:
#     k_r = k_r.contiguous().view(w, w, cross_attn_layers.cross_attn.num_heads, head_dim)

# # compute attn weight
# attn_feat = torch.einsum("wnec,vnec->newv", q, k)  # NxExWxW'

# # add positional terms
# if pos_enc is not None:
#     # 0.3 s
#     attn_feat_pos = torch.einsum("wnec,wvec->newv", q, k_r)  # NxExWxW'
#     attn_pos_feat = torch.einsum("vnec,wvec->newv", k, q_r)  # NxExWxW'

#     # 0.1 s
#     attn = attn_feat + attn_feat_pos + attn_pos_feat
# else:
#     attn = attn_feat

# assert list(attn.size()) == [bsz, cross_attn_layers.cross_attn.num_heads, w, w]

# # apply attn mask
# if attn_mask is not None:
#     attn_mask = attn_mask[None, None, ...]
#     attn += attn_mask

# # raw attn
# raw_attn = attn

# # softmax
# attn = F.softmax(attn, dim=-1)

# # compute v, equivalent to einsum('',attn,v),
# # need to do this because apex does not support einsum when precision is mixed
# v_o = torch.bmm(
#     attn.view(bsz * cross_attn_layers.cross_attn.num_heads, w, w),
#     v.permute(1, 2, 0, 3).view(
#         bsz * cross_attn_layers.cross_attn.num_heads, w, head_dim
#     ),
# )  # NxExWxW', W'xNxExC -> NExWxC
# assert list(v_o.size()) == [bsz * cross_attn_layers.cross_attn.num_heads, w, head_dim]
# v_o = (
#     v_o.reshape(bsz, cross_attn_layers.cross_attn.num_heads, w, head_dim)
#     .permute(2, 0, 1, 3)
#     .reshape(w, bsz, embed_dim)
# )
# v_o = F.linear(
#     v_o,
#     cross_attn_layers.cross_attn.out_proj.weight,
#     cross_attn_layers.cross_attn.out_proj.bias,
# )

# # average attention weights over heads
# attn = attn.sum(dim=1) / cross_attn_layers.cross_attn.num_heads

# # raw attn
# raw_attn = raw_attn.sum(dim=1)

# # %% debug regression layer
# from module.regression_head import build_regression_head

# regression_head = build_regression_head(config)
# regression_head = regression_head.to(device)

# bs, _, h, w = inputs.left.size()
# output = {}

# # compute scale
# if inputs.sampled_cols is not None:
#     scale = inputs.left.size(-1) / float(inputs.sampled_cols.size(-1))
# else:
#     scale = 1.0

# # normalize attention to 0-1
# if regression_head.ot:
#     # optimal transport
#     attn_ot = regression_head._optimal_transport(attn_weight, 10)
# else:
#     # softmax
#     attn_ot = regression_head._softmax(attn_weight)

# # compute relative response (RR) at ground truth location
# if inputs.disp is not None:
#     # find ground truth response (gt_response) and location (target)
#     output["gt_response"], target = regression_head._compute_gt_location(
#         scale,
#         inputs.sampled_cols,
#         inputs.sampled_rows,
#         attn_ot[..., :-1, :-1],
#         inputs.disp,
#     )
# else:
#     output["gt_response"] = None

# # compute relative response (RR) at occluded location
# if inputs.occ_mask is not None:
#     # handle occlusion
#     occ_mask = inputs.occ_mask  # [B, 360, 640]
#     occ_mask_right = inputs.occ_mask_right  # [B, 360, 640]
#     if inputs.sampled_cols is not None:
#         occ_mask = batched_index_select(occ_mask, 2, inputs.sampled_cols)
#         occ_mask_right = batched_index_select(occ_mask_right, 2, inputs.sampled_cols)
#     if inputs.sampled_rows is not None:
#         occ_mask = batched_index_select(occ_mask, 1, inputs.sampled_rows)
#         occ_mask_right = batched_index_select(occ_mask_right, 1, inputs.sampled_rows)

#     output["gt_response_occ_left"] = attn_ot[..., :-1, -1][occ_mask]
#     output["gt_response_occ_right"] = attn_ot[..., -1, :-1][occ_mask_right]
# else:
#     output["gt_response_occ_left"] = None
#     output["gt_response_occ_right"] = None
#     occ_mask = inputs.occ_mask

# # regress low res disparity
# pos_shift = regression_head._compute_unscaled_pos_shift(
#     attn_weight.shape[2], attn_weight.device
# )  # NxHxW_leftxW_right
# disp_pred_low_res, matched_attn = regression_head._compute_low_res_disp(
#     pos_shift, attn_ot[..., :-1, :-1], occ_mask
# )
# # regress low res occlusion
# occ_pred_low_res = regression_head._compute_low_res_occ(matched_attn)

# # with open('attn_weight.dat', 'wb') as f:
# #     torch.save(attn_ot[0], f)
# # with open('target.dat', 'wb') as f:
# #     torch.save(target, f)

# # upsample and context adjust
# if inputs.sampled_cols is not None:
#     output["disp_pred"], output["disp_pred_low_res"], output["occ_pred"] = (
#         regression_head._upsample(inputs, disp_pred_low_res, occ_pred_low_res, scale)
#     )
# else:
#     output["disp_pred"] = disp_pred_low_res
#     output["occ_pred"] = occ_pred_low_res
# # %%
# _, _, h, w = inputs.left.size()
# # scale disparity
# disp_pred_attn = disp_pred_low_res * scale

# # upsample
# disp_pred = F.interpolate(
#     disp_pred_attn[:, None], size=(h, w), mode="nearest"
# )  # N x 1 x H x W
# occ_pred = F.interpolate(
#     occ_pred_low_res[:, None], size=(h, w), mode="nearest"
# )  # N x 1 x H x W

# if regression_head.cal is not None:
#     # normalize disparity
#     eps = 1e-6
#     mean_disp_pred = disp_pred.mean()
#     std_disp_pred = disp_pred.std() + eps
#     disp_pred_normalized = (disp_pred - mean_disp_pred) / std_disp_pred

#     # normalize occlusion mask
#     occ_pred_normalized = (occ_pred - 0.5) / 0.5

#     disp_pred_normalized, occ_pred = regression_head.cal(
#         disp_pred_normalized, occ_pred_normalized, inputs.left
#     )  # N x H x W

#     disp_pred_final = disp_pred_normalized * std_disp_pred + mean_disp_pred

# # context adjustment layer
# feat_cal = regression_head.cal.in_conv(
#     torch.cat([disp_pred_normalized, inputs.left], dim=1)
# )

# disp_final, occ_final = regression_head()
