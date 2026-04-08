from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


COLOR_MAP = {
    "True": "tab:blue",
    "Autodiff": "tab:orange",
    "OurModel": "tab:green",
    "Observed": "tab:blue",
}



def _to_BTG(Y):
    Y = Y.detach().cpu().numpy()
    if Y.ndim == 2:
        Y = Y[None, ...]
    return Y



def plot_seed_trajectories(
    t_span,
    Y_obs,
    Y_true,
    predictions,
    title_prefix="",
    save_path=None,
    show=True,
):
    t = t_span.detach().cpu().numpy()
    Yo = _to_BTG(Y_obs)
    Yt = _to_BTG(Y_true)
    pred_np = {name: _to_BTG(Y) for name, Y in predictions.items()}

    B, _, d = Yo.shape
    nd = min(3, d)
    fig, axes = plt.subplots(1, nd, figsize=(5 * nd, 4), sharex=True)
    if nd == 1:
        axes = [axes]

    for j in range(nd):
        ax = axes[j]
        for b in range(B):
            ls = "-" if b == 0 else "--"
            alpha_obs = 0.75 if b == 0 else 0.45
            ax.plot(t, Yt[b, :, j], color=COLOR_MAP["True"], linestyle=ls, label=f"True (b={b})")
            for name, arr in pred_np.items():
                ax.plot(t, arr[b, :, j], linestyle=ls, label=f"{name} (b={b})")
            ax.scatter(t, Yo[b, :, j], s=10, color=COLOR_MAP["Observed"], alpha=alpha_obs, label=f"Y_obs (b={b})")

        ax.set_xlabel("t")
        ax.set_ylabel(f"dim {j}")
        ax.set_title(f"{title_prefix} dim {j}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig



def plot_seed_losses(loss_histories, checkpoints=None, title_prefix="", save_path=None, show=True):
    fig = plt.figure(figsize=(7, 4))
    for name, history in loss_histories.items():
        x = np.arange(1, len(history) + 1)
        plt.plot(x, history, label=name)

    if checkpoints:
        for cp in checkpoints:
            if cp > 0:
                plt.axvline(cp, linestyle="--", linewidth=1, alpha=0.4)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} loss curves")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
