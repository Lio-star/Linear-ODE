from __future__ import annotations

import time

import torch

from utils.frechet import expm_pade13_prepare, expm_frechet_from_cache


def run_ou_nll_A_xstar(
    X0,
    Y_obs,
    t_span,
    Mask,
    epochs=5000,
    lr_A=3e-3,
    lr_xs=3e-3,
    lr_logsig=1e-3,
    K_time=None,
    eps=1e-8,
    print_every=1000,
    seed=0,
    do_print=True,
    init_A=None,
    init_log_sigma2=None,
):
    """
    Fit the same linear dynamical system as `our_model`, but train it using
    Gaussian negative log-likelihood with a learned global noise variance.

    Returns a dict aligned with the existing project output format.
    """
    assert X0.dim() == 2 and Y_obs.dim() == 3

    B, T, G = Y_obs.shape
    device, dtype = Y_obs.device, Y_obs.dtype

    X0 = X0.to(device=device, dtype=dtype)
    Y_obs = Y_obs.to(device=device, dtype=dtype)
    t_span = t_span.to(device=device, dtype=dtype)
    Mask = Mask.to(device=device)
    Mask_f = Mask.to(device=device, dtype=dtype)

    def apply_mask(A):
        return A * Mask_f

    if K_time is None:
        K_time = min(1, T)
    else:
        K_time = min(K_time, T)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if init_A is None:
        A = apply_mask(0.01 * torch.randn(G, G, device=device, dtype=dtype))
    else:
        A = apply_mask(init_A.clone().to(device=device, dtype=dtype))

    tail_k = min(10, T)
    x_star = (
        Y_obs[:, -tail_k:, :]
        .mean(dim=(0, 1))
        .detach()
        .clone()
        .to(device=device, dtype=dtype)
    )

    with torch.no_grad():
        E_all0 = torch.matrix_exp(A.unsqueeze(0) * t_span.view(T, 1, 1))
        Y0_init = torch.einsum("tij,bj->bti", E_all0, (X0 - x_star)) + x_star
        resid0 = Y0_init - Y_obs
        sigma2_init = resid0.pow(2).mean().clamp_min(
            torch.tensor(1e-6, device=device, dtype=dtype)
        )

    if init_log_sigma2 is None:
        log_sigma2 = torch.log(sigma2_init).detach().clone()
    else:
        log_sigma2 = torch.tensor(init_log_sigma2, device=device, dtype=dtype)

    beta1, beta2 = 0.9, 0.999
    adam_eps = 1e-8

    mA = torch.zeros_like(A)
    vA = torch.zeros_like(A)
    mx = torch.zeros_like(x_star)
    vx = torch.zeros_like(x_star)
    ms = torch.zeros_like(log_sigma2)
    vs = torch.zeros_like(log_sigma2)

    I = torch.eye(G, device=device, dtype=dtype)
    eps_t = torch.tensor(eps, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    loss_hist = []

    for ep in range(1, epochs + 1):
        idx_list = torch.randint(1, T, (K_time,), device="cpu").tolist()

        gradA = torch.zeros_like(A)
        gradx = torch.zeros_like(x_star)
        grad_logsig = torch.zeros_like(log_sigma2)
        loss_acc = torch.zeros((), device=device, dtype=dtype)

        A_eff = apply_mask(A)
        sigma2 = torch.exp(log_sigma2).clamp_min(eps_t)

        idx = torch.tensor(idx_list, device=device)
        t_vec = t_span.index_select(0, idx)
        X_batch = t_vec[:, None, None] * A_eff[None, :, :]
        R_batch = torch.matrix_exp(X_batch)

        for k in range(len(idx_list)):
            i = int(idx[k].item())
            t = t_vec[k]

            if float(torch.abs(t).item()) < 1e-15:
                continue

            R = R_batch[k]
            _, cache = expm_pade13_prepare(X_batch[k])

            D = (X0 - x_star).T
            C = (Y_obs[:, i, :] - x_star).T

            R_D = R @ D
            r_mat = R_D - C
            g_mat = r_mat / sigma2

            E_total = t * (D @ g_mat.T)
            L_total = expm_frechet_from_cache(cache, E_total)
            gradA = gradA + L_total.T

            Bmat = I - R
            gradx = gradx + (Bmat.T @ g_mat).sum(dim=1)

            sq_per_batch = r_mat.pow(2).sum(dim=0)
            grad_logsig = grad_logsig + (
                0.5 * G - 0.5 * sq_per_batch / sigma2
            ).sum()

            nll_each = 0.5 * (sq_per_batch / sigma2 + G * log_sigma2)
            loss_acc = loss_acc + nll_each.sum() / G

        denom = max(1, B * K_time)

        gradA = gradA / (denom * G)
        gradx = gradx / (denom * G)
        grad_logsig = grad_logsig / (denom * G)

        mA = beta1 * mA + (1 - beta1) * gradA
        vA = beta2 * vA + (1 - beta2) * (gradA * gradA)
        mA_hat = mA / (1 - beta1**ep)
        vA_hat = vA / (1 - beta2**ep)
        A = A - lr_A * (mA_hat / (torch.sqrt(vA_hat) + adam_eps))
        A = apply_mask(A)

        mx = beta1 * mx + (1 - beta1) * gradx
        vx = beta2 * vx + (1 - beta2) * (gradx * gradx)
        mx_hat = mx / (1 - beta1**ep)
        vx_hat = vx / (1 - beta2**ep)
        x_star = x_star - lr_xs * (mx_hat / (torch.sqrt(vx_hat) + adam_eps))

        ms = beta1 * ms + (1 - beta1) * grad_logsig
        vs = beta2 * vs + (1 - beta2) * (grad_logsig * grad_logsig)
        ms_hat = ms / (1 - beta1**ep)
        vs_hat = vs / (1 - beta2**ep)
        log_sigma2 = log_sigma2 - lr_logsig * (
            ms_hat / (torch.sqrt(vs_hat) + adam_eps)
        )

        if do_print and (print_every is not None) and (
            ep % print_every == 0 or ep == 1
        ):
            sigma2_now = torch.exp(log_sigma2).item()
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(
                f"epoch {ep:04d} | OU-NLL {(loss_acc / denom).item():.4e} | "
                f"sigma2 {sigma2_now:.4e} | "
                f"||A||_F {torch.linalg.norm(A).item():.3e} | "
                f"||x*|| {torch.linalg.norm(x_star).item():.3e}"
            )

        loss_hist.append(float((loss_acc / denom).item()))

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_sec = time.time() - t0

    with torch.no_grad():
        E_all = torch.matrix_exp(A.unsqueeze(0) * t_span.view(T, 1, 1))
        Y_pred = torch.einsum("tij,bj->bti", E_all, (X0 - x_star)) + x_star
        sigma2_hat = float(torch.exp(log_sigma2).item())

    return {
        "name": "ou_nll",
        "display_name": "OU-NLL",
        "time_sec": total_sec,
        "final_loss": loss_hist[-1] if loss_hist else None,
        "loss_history": loss_hist,
        "A_hat": A,
        "x_star_hat": x_star,
        "sigma2_hat": sigma2_hat,
        "Y_pred": Y_pred,
    }


def compute_ou_nll_epoch0_loss(X0, Y_obs, t_span, Mask, init_A):
    """
    Compute the OU-NLL objective at epoch 0 using the shared initial A and the
    same x_star initialization rule as training.
    """
    del Mask  # kept for interface symmetry

    device = Y_obs.device
    dtype = Y_obs.dtype
    G = Y_obs.shape[2]

    with torch.no_grad():
        tail_k = min(10, Y_obs.shape[1])
        x_star0 = (
            Y_obs[:, -tail_k:, :]
            .mean(dim=(0, 1))
            .detach()
            .clone()
            .to(device=device, dtype=dtype)
        )

        A0 = init_A.to(device=device, dtype=dtype)
        E0 = torch.matrix_exp(A0.unsqueeze(0) * t_span.view(t_span.shape[0], 1, 1))
        Y0 = torch.einsum("tij,bj->bti", E0, (X0 - x_star0)) + x_star0

        resid0 = Y0 - Y_obs
        sigma2_0 = resid0.pow(2).mean().clamp_min(
            torch.tensor(1e-6, device=device, dtype=dtype)
        )
        loss0 = (
            0.5 * (resid0.pow(2).sum(dim=2) / sigma2_0 + G * torch.log(sigma2_0))
        ).mean().item() / G

    return float(loss0), Y0, float(sigma2_0.item())
