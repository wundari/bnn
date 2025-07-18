# %% load necessary modules
import numpy as np

from config.config import BNNconfig

import seaborn as sns
import matplotlib.pyplot as plt

import os
import json
from jaxtyping import Float


# %%
class GA_LearningCurve:
    """
    Class for analyzing and plotting learning curves for different binocular
    interactions.

    It gathers training and validation losses and accuracies across multiple
    seeds, and provides methods to plot these metrics.
    """

    def __init__(self, config: BNNconfig):

        self.config = config

        # folder location where training loss is stored
        self.experiment_dir = (
            f"run/{config.dataset}/bino_interaction_{config.binocular_interaction}"
        )

        # create folder for saving plots
        self.plot_dir = f"{self.experiment_dir}/plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # create folder for saving group analysis
        self.group_dir = f"run/{config.dataset}/bino_interaction_group"
        if not os.path.exists(self.group_dir):
            os.makedirs(self.group_dir)

        # create folder for saving group analysis plots
        self.group_plot_dir = f"{self.group_dir}/plots"
        if not os.path.exists(self.group_plot_dir):
            os.makedirs(self.group_plot_dir)

        # save config
        self.save_config()

    def save_config(self):
        config_dict = self.config.to_dict()
        with open(f"{self.experiment_dir}/config.json", "w") as f:
            json.dump(config_dict, f)

    def update_bino_interaction(self, interaction: str) -> None:
        """
        Update the binocular interaction in the config and experiment directory.

        Args:
            interaction (str): The new binocular interaction to set.
        """
        # update the binocular interaction in the config
        self.config.binocular_interaction = interaction

        # update the experiment and plot directories based on the new interaction
        self.experiment_dir = (
            f"run/{self.config.dataset}/bino_interaction_{interaction}"
        )
        self.plot_dir = f"{self.experiment_dir}/plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # save config
        self.save_config()

        print(
            "Update binocular interaction from "
            + f"{self.config.binocular_interaction} to {interaction}.\n"
            + f"Experiment directory: {self.experiment_dir}\n"
            + f"Plot directory: {self.plot_dir}"
        )

    def get_train_loss(self) -> Float[np.ndarray, "n_seed n_step"]:
        """
        gather training losses over all seeds for a given binocular interaction.

        Returns:
            losses<np.ndarray [n_seed, n_step]> : train losses for each seed
        """

        losses = []
        for seed in self.config.seed_to_analyse:

            # load training loss
            loss = np.load(f"{self.experiment_dir}/{seed}/losses_train.npy")
            losses.append(loss)

        # list to array
        losses = np.array(losses)

        return losses

    def get_train_acc(self) -> Float[np.ndarray, "n_seed n_step"]:
        """
        gather training accuracies over all seeds for a given binocular interaction.

        Returns:
            accs<np.ndarray [n_seed, n_step]> : train accuracies
                for each seed
        """

        accs = []
        for seed in self.config.seed_to_analyse:

            # load training accuracy
            loss = np.load(f"{self.experiment_dir}/{seed}/accs_train.npy")
            accs.append(loss)

        # list to array
        accs = np.array(accs)

        return accs

    def get_val_loss(self) -> Float[np.ndarray, "n_seed n_step"]:
        """
        gather validation losses over all seeds for a given binocular interaction.

        Returns:
            losses<np.ndarray [n_seed, n_step]> : validation losses
                for each seed
        """

        losses = []
        for seed in self.config.seed_to_analyse:

            # load validation loss
            loss = np.load(f"{self.experiment_dir}/{seed}/losses_val.npy")
            losses.append(loss)

        # list to array
        losses = np.array(losses)

        return losses

    def get_val_acc(self) -> Float[np.ndarray, "n_seed n_step"]:
        """
        gather validation accuracies over all seeds for a given binocular interaction.

        Returns:
            accs<np.ndarray [n_seed, n_step]> : validation accuracies
                for each seed
        """

        accs = []
        for seed in self.config.seed_to_analyse:

            # load validation accuracy
            loss = np.load(f"{self.experiment_dir}/{seed}/accs_val.npy")
            accs.append(loss)

        # list to array
        accs = np.array(accs)

        return accs

    def plotLine_loss_all_seeds(
        self,
        train_losses: Float[np.ndarray, "n_seed n_step"],
        val_losses: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot training and validation loss across all seeds
        of a given binocular interaction.

        Args:
            train_losses (Float[np.ndarray, "n_seed n_step"]): Training losses
                for each seed.

            val_losses (Float[np.ndarray, "n_seed n_step"]): Validation losses
                for each seed.

            save_flag <bool>: If True, save the plot to a file.
                Default is False.
        Returns:
            None

        """

        # average losses across seeds
        train_loss_avg = np.mean(train_losses, axis=0)  # [n_steps]
        train_loss_std = np.std(train_losses, axis=0)
        val_loss_avg = np.mean(val_losses, axis=0)
        val_loss_std = np.std(val_losses, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Train and validation losses all seeds \n({self.config.binocular_interaction})",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#00CED1"]
        axes.plot(train_loss_avg, linewidth=2, color=colors[0], label="Train")
        axes.fill_between(
            np.arange(len(train_loss_avg)),
            train_loss_avg - train_loss_std,
            train_loss_avg + train_loss_std,
            color=colors[0],
            alpha=0.2,
        )
        axes.plot(val_loss_avg, linewidth=2, color=colors[1], label="Val")
        axes.fill_between(
            np.arange(len(val_loss_avg)),
            val_loss_avg - val_loss_std,
            val_loss_avg + val_loss_std,
            color=colors[1],
            alpha=0.2,
        )

        x_low = 0
        x_up = 225
        x_step = 50
        y_low = 20
        y_up = 61
        y_step = 10

        axes.set_xlabel("Steps (x 100)")
        axes.set_ylabel("L1-loss")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        plt.legend(
            loc="upper right",
            fontsize=20,
            frameon=False,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save plot
        if save_flag:
            plt.savefig(
                f"{self.plot_dir}/plotLine_loss_avg.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_train_loss_all_interactions(
        self,
        train_loss_default: Float[np.ndarray, "n_seed n_step"],
        train_loss_bem: Float[np.ndarray, "n_seed n_step"],
        train_loss_cmm: Float[np.ndarray, "n_seed n_step"],
        train_loss_sum_diff: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot training losses for all binocular interactions.
        Args:
            train_loss_default (Float[np.ndarray, "n_seed n_step"]): train
                losses for default interaction.

            train_loss_bem (Float[np.ndarray, "n_seed n_step"]): train
                losses for BEM interaction.

            train_loss_cmm (Float[np.ndarray, "n_seed n_step"]): train
                losses for CMM interaction.

            train_loss_sum_diff (Float[np.ndarray, "n_seed n_step"]): train
                losses for sum_diff interaction.

            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average losses
        train_loss_default_avg = np.mean(train_loss_default, axis=0)
        train_loss_default_std = np.std(train_loss_default, axis=0)
        train_loss_bem_avg = np.mean(train_loss_bem, axis=0)
        train_loss_bem_std = np.std(train_loss_bem, axis=0)
        train_loss_cmm_avg = np.mean(train_loss_cmm, axis=0)
        train_loss_cmm_std = np.std(train_loss_cmm, axis=0)
        train_loss_sum_diff_avg = np.mean(train_loss_sum_diff, axis=0)
        train_loss_sum_diff_std = np.std(train_loss_sum_diff, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Training losses\nall binocular_interaction",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        axes.plot(train_loss_default_avg, linewidth=2, color=colors[0], label=labels[0])
        axes.fill_between(
            np.arange(len(train_loss_default_avg)),
            train_loss_default_avg - train_loss_default_std,
            train_loss_default_avg + train_loss_default_std,
            color=colors[0],
            alpha=0.2,
        )

        axes.plot(train_loss_bem_avg, linewidth=2, color=colors[1], label=labels[1])
        axes.fill_between(
            np.arange(len(train_loss_bem_avg)),
            train_loss_bem_avg - train_loss_bem_std,
            train_loss_bem_avg + train_loss_bem_std,
            color=colors[1],
            alpha=0.2,
        )

        axes.plot(train_loss_cmm_avg, linewidth=2, color=colors[2], label=labels[2])
        axes.fill_between(
            np.arange(len(train_loss_cmm_avg)),
            train_loss_cmm_avg - train_loss_cmm_std,
            train_loss_cmm_avg + train_loss_cmm_std,
            color=colors[2],
            alpha=0.2,
        )

        axes.plot(
            train_loss_sum_diff_avg, linewidth=2, color=colors[3], label=labels[3]
        )
        axes.fill_between(
            np.arange(len(train_loss_sum_diff_avg)),
            train_loss_sum_diff_avg - train_loss_sum_diff_std,
            train_loss_sum_diff_avg + train_loss_sum_diff_std,
            color=colors[3],
            alpha=0.2,
        )

        x_low = 0
        x_up = 225
        x_step = 50
        y_low = 20
        y_up = 61
        y_step = 10

        axes.set_xlabel("Steps (x 100)")
        axes.set_ylabel("L1-loss")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        plt.legend(
            loc="upper right",
            fontsize=20,
            frameon=False,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save plot
        if save_flag:
            plt.savefig(
                f"{self.group_plot_dir}/plotLine_train_loss_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_val_loss_all_interactions(
        self,
        val_loss_default: Float[np.ndarray, "n_seed n_step"],
        val_loss_bem: Float[np.ndarray, "n_seed n_step"],
        val_loss_cmm: Float[np.ndarray, "n_seed n_step"],
        val_loss_sum_diff: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot validation losses for all binocular interactions.
        Args:
            val_loss_default (Float[np.ndarray, "n_seed n_step"]): Validation losses for default interaction.
            val_loss_bem (Float[np.ndarray, "n_seed n_step"]): Validation losses for BEM interaction.
            val_loss_cmm (Float[np.ndarray, "n_seed n_step"]): Validation losses for CMM interaction.
            val_loss_sum_diff (Float[np.ndarray, "n_seed n_step"]): Validation losses for sum_diff interaction.
            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average losses
        val_loss_default_avg = np.mean(val_loss_default, axis=0)
        val_loss_default_std = np.std(val_loss_default, axis=0)
        val_loss_bem_avg = np.mean(val_loss_bem, axis=0)
        val_loss_bem_std = np.std(val_loss_bem, axis=0)
        val_loss_cmm_avg = np.mean(val_loss_cmm, axis=0)
        val_loss_cmm_std = np.std(val_loss_cmm, axis=0)
        val_loss_sum_diff_avg = np.mean(val_loss_sum_diff, axis=0)
        val_loss_sum_diff_std = np.std(val_loss_sum_diff, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Validation losses\nall binocular_interaction",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        axes.plot(val_loss_default_avg, linewidth=2, color=colors[0], label=labels[0])
        axes.fill_between(
            np.arange(len(val_loss_default_avg)),
            val_loss_default_avg - val_loss_default_std,
            val_loss_default_avg + val_loss_default_std,
            color=colors[0],
            alpha=0.2,
        )

        axes.plot(val_loss_bem_avg, linewidth=2, color=colors[1], label=labels[1])
        axes.fill_between(
            np.arange(len(val_loss_bem_avg)),
            val_loss_bem_avg - val_loss_bem_std,
            val_loss_bem_avg + val_loss_bem_std,
            color=colors[1],
            alpha=0.2,
        )

        axes.plot(val_loss_cmm_avg, linewidth=2, color=colors[2], label=labels[2])
        axes.fill_between(
            np.arange(len(val_loss_cmm_avg)),
            val_loss_cmm_avg - val_loss_cmm_std,
            val_loss_cmm_avg + val_loss_cmm_std,
            color=colors[2],
            alpha=0.2,
        )

        axes.plot(val_loss_sum_diff_avg, linewidth=2, color=colors[3], label=labels[3])
        axes.fill_between(
            np.arange(len(val_loss_sum_diff_avg)),
            val_loss_sum_diff_avg - val_loss_sum_diff_std,
            val_loss_sum_diff_avg + val_loss_sum_diff_std,
            color=colors[3],
            alpha=0.2,
        )

        x_low = 0
        x_up = 225
        x_step = 50
        y_low = 20
        y_up = 61
        y_step = 10

        axes.set_xlabel("Steps (x 100)")
        axes.set_ylabel("L1-loss")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        plt.legend(
            loc="upper right",
            fontsize=20,
            frameon=False,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save plot
        if save_flag:
            plt.savefig(
                f"{self.group_plot_dir}/plotLine_val_loss_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_acc(
        self,
        train_accs: Float[np.ndarray, "n_seed n_step"],
        val_accs: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot training and validation accuracy.

        Args:
            train_accs (Float[np.ndarray, "n_seed n_step"]): Training accuracies for each seed.
            val_accs (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for each seed.
            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average accuracies
        train_acc_avg = np.mean(train_accs, axis=0)
        train_acc_std = np.std(train_accs, axis=0)
        val_acc_avg = np.mean(val_accs, axis=0)
        val_acc_std = np.std(val_accs, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            f"Train and validation accuracies all seeds \n({self.config.binocular_interaction})",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#00CED1"]
        axes.plot(train_acc_avg, linewidth=2, color=colors[0], label="Train")
        axes.fill_between(
            np.arange(len(train_acc_avg)),
            train_acc_avg - train_acc_std,
            train_acc_avg + train_acc_std,
            color=colors[0],
            alpha=0.2,
        )
        axes.plot(val_acc_avg, linewidth=2, color=colors[1], label="Val")
        axes.fill_between(
            np.arange(len(val_acc_avg)),
            val_acc_avg - val_acc_std,
            val_acc_avg + val_acc_std,
            color=colors[1],
            alpha=0.2,
        )

        x_low = 0
        x_up = 225
        x_step = 50
        y_low = 0.0
        y_up = 0.61
        y_step = 0.1

        axes.set_xlabel("Steps (x 100)")
        axes.set_ylabel("3-pix accuracy")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        plt.legend(
            loc="lower right",
            fontsize=20,
            frameon=False,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save plot
        if save_flag:
            plt.savefig(
                f"{self.plot_dir}/plotLine_acc_avg.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_train_acc_all_interactions(
        self,
        train_acc_default: Float[np.ndarray, "n_seed n_step"],
        train_acc_bem: Float[np.ndarray, "n_seed n_step"],
        train_acc_cmm: Float[np.ndarray, "n_seed n_step"],
        train_acc_sum_diff: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot validation accuracies for all binocular interactions.
        Args:
            train_acc_default (Float[np.ndarray, "n_seed n_step"]): train accuracies for default interaction.
            train_acc_bem (Float[np.ndarray, "n_seed n_step"]): train accuracies for BEM interaction.
            train_acc_cmm (Float[np.ndarray, "n_seed n_step"]): train accuracies for CMM interaction.
            train_acc_sum_diff (Float[np.ndarray, "n_seed n_step"]): train accuracies for sum_diff interaction.
            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average losses
        train_acc_default_avg = np.mean(train_acc_default, axis=0)
        train_acc_default_std = np.std(train_acc_default, axis=0)
        train_acc_bem_avg = np.mean(train_acc_bem, axis=0)
        train_acc_bem_std = np.std(train_acc_bem, axis=0)
        train_acc_cmm_avg = np.mean(train_acc_cmm, axis=0)
        train_acc_cmm_std = np.std(train_acc_cmm, axis=0)
        train_acc_sum_diff_avg = np.mean(train_acc_sum_diff, axis=0)
        train_acc_sum_diff_std = np.std(train_acc_sum_diff, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Train accuracies\nall binocular_interaction",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        axes.plot(train_acc_default_avg, linewidth=2, color=colors[0], label=labels[0])
        axes.fill_between(
            np.arange(len(train_acc_default_avg)),
            train_acc_default_avg - train_acc_default_std,
            train_acc_default_avg + train_acc_default_std,
            color=colors[0],
            alpha=0.2,
        )

        axes.plot(train_acc_bem_avg, linewidth=2, color=colors[1], label=labels[1])
        axes.fill_between(
            np.arange(len(train_acc_bem_avg)),
            train_acc_bem_avg - train_acc_bem_std,
            train_acc_bem_avg + train_acc_bem_std,
            color=colors[1],
            alpha=0.2,
        )

        axes.plot(train_acc_cmm_avg, linewidth=2, color=colors[2], label=labels[2])
        axes.fill_between(
            np.arange(len(train_acc_cmm_avg)),
            train_acc_cmm_avg - train_acc_cmm_std,
            train_acc_cmm_avg + train_acc_cmm_std,
            color=colors[2],
            alpha=0.2,
        )

        axes.plot(train_acc_sum_diff_avg, linewidth=2, color=colors[3], label=labels[3])
        axes.fill_between(
            np.arange(len(train_acc_sum_diff_avg)),
            train_acc_sum_diff_avg - train_acc_sum_diff_std,
            train_acc_sum_diff_avg + train_acc_sum_diff_std,
            color=colors[3],
            alpha=0.2,
        )

        x_low = 0
        x_up = 225
        x_step = 50
        y_low = 0.0
        y_up = 0.61
        y_step = 0.1

        axes.set_xlabel("Steps (x 100)")
        axes.set_ylabel("3-pix accuracy")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        plt.legend(
            loc="lower right",
            fontsize=20,
            frameon=False,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save plot
        if save_flag:
            plt.savefig(
                f"{self.group_plot_dir}/plotLine_train_acc_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )

    def plotLine_val_acc_all_interactions(
        self,
        val_acc_default: Float[np.ndarray, "n_seed n_step"],
        val_acc_bem: Float[np.ndarray, "n_seed n_step"],
        val_acc_cmm: Float[np.ndarray, "n_seed n_step"],
        val_acc_sum_diff: Float[np.ndarray, "n_seed n_step"],
        save_flag: bool = False,
    ) -> None:
        """
        Plot validation accuracies for all binocular interactions.
        Args:
            val_acc_default (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for default interaction.
            val_acc_bem (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for BEM interaction.
            val_acc_cmm (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for CMM interaction.
            val_acc_sum_diff (Float[np.ndarray, "n_seed n_step"]): Validation accuracies for sum_diff interaction.
            save_flag (bool): If True, save the plot to a file. Default is False.
        Returns:
            None
        """

        # average losses
        val_acc_default_avg = np.mean(val_acc_default, axis=0)
        val_acc_default_std = np.std(val_acc_default, axis=0)
        val_acc_bem_avg = np.mean(val_acc_bem, axis=0)
        val_acc_bem_std = np.std(val_acc_bem, axis=0)
        val_acc_cmm_avg = np.mean(val_acc_cmm, axis=0)
        val_acc_cmm_std = np.std(val_acc_cmm, axis=0)
        val_acc_sum_diff_avg = np.mean(val_acc_sum_diff, axis=0)
        val_acc_sum_diff_std = np.std(val_acc_sum_diff, axis=0)

        # plotting
        sns.set_theme()
        sns.set_theme(context="paper", style="white", font_scale=3, palette="deep")

        figsize = (10, 6)
        n_row = 1
        n_col = 1

        fig, axes = plt.subplots(
            nrows=n_row, ncols=n_col, figsize=figsize, sharex=True, sharey=True
        )

        fig.text(
            0.5,
            1.0,
            "Validation accuracies\nall binocular_interaction",
            ha="center",
        )
        # fig.text(-0.05, 0.5, "L1 loss", va="center", rotation=90)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)

        colors = ["#333333", "#6a5acd", "#B22222", "#00CED1"]
        labels = ["Concat", "BEM", "CMM", "Sum Diff"]

        axes.plot(val_acc_default_avg, linewidth=2, color=colors[0], label=labels[0])
        axes.fill_between(
            np.arange(len(val_acc_default_avg)),
            val_acc_default_avg - val_acc_default_std,
            val_acc_default_avg + val_acc_default_std,
            color=colors[0],
            alpha=0.2,
        )

        axes.plot(val_acc_bem_avg, linewidth=2, color=colors[1], label=labels[1])
        axes.fill_between(
            np.arange(len(val_acc_bem_avg)),
            val_acc_bem_avg - val_acc_bem_std,
            val_acc_bem_avg + val_acc_bem_std,
            color=colors[1],
            alpha=0.2,
        )

        axes.plot(val_acc_cmm_avg, linewidth=2, color=colors[2], label=labels[2])
        axes.fill_between(
            np.arange(len(val_acc_cmm_avg)),
            val_acc_cmm_avg - val_acc_cmm_std,
            val_acc_cmm_avg + val_acc_cmm_std,
            color=colors[2],
            alpha=0.2,
        )

        axes.plot(val_acc_sum_diff_avg, linewidth=2, color=colors[3], label=labels[3])
        axes.fill_between(
            np.arange(len(val_acc_sum_diff_avg)),
            val_acc_sum_diff_avg - val_acc_sum_diff_std,
            val_acc_sum_diff_avg + val_acc_sum_diff_std,
            color=colors[3],
            alpha=0.2,
        )

        x_low = 0
        x_up = 225
        x_step = 50
        y_low = 0.0
        y_up = 0.61
        y_step = 0.1

        axes.set_xlabel("Steps (x 100)")
        axes.set_ylabel("3-pix accuracy")
        axes.set_xticks(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_xticklabels(np.round(np.arange(x_low, x_up, x_step), 2))
        axes.set_yticks(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_yticklabels(np.round(np.arange(y_low, y_up, y_step), 2))
        axes.set_xlim(x_low, x_up)
        axes.set_ylim(y_low, y_up)

        plt.legend(
            loc="lower right",
            fontsize=20,
            frameon=False,
        )

        # Hide the right and top spines
        axes.spines["right"].set_visible(False)
        axes.spines["top"].set_visible(False)
        # Only show ticks on the left and bottom spines
        axes.yaxis.set_ticks_position("left")
        axes.xaxis.set_ticks_position("bottom")

        # save plot
        if save_flag:
            plt.savefig(
                f"{self.group_plot_dir}/plotLine_val_acc_all_interactions.pdf",
                dpi=600,
                bbox_inches="tight",
            )
