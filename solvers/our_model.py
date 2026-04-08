import time
import torch
import torch.nn.functional as F

from utils.frechet import expm_frechet_alg64_pade13, expm_pade13_prepare, expm_frechet_from_cache



def run_ourmethod_learn_A_xstar(
    X0,
    Y_obs,
    t_span,
    Mask,
    epochs=5000,
    lr_A=3e-3,
    lr_xs=3e-3,
    K_time=None,
    delta=0.07,
    eps=1e-12,
    print_every=1000,
    seed=0,
    do_print=True,
    init_A=None,
):
    assert X0.dim() == 2 and Y_obs.dim() == 3
    B, T, G = Y_obs.shape
    device, dtype = Y_obs.device, Y_obs.dtype

    X0 = X0.to(device=device, dtype=dtype)
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

    x_star = Y_obs[:, -10:, :].mean(dim=(0, 1)).detach().clone().to(device=device, dtype=dtype)

    beta1, beta2 = 0.9, 0.999
    mA = torch.zeros_like(A)
    vA = torch.zeros_like(A)
    mx = torch.zeros_like(x_star)
    vx = torch.zeros_like(x_star)
    adam_eps = 1e-8

    I = torch.eye(G, device=device, dtype=dtype)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    delta_t = torch.tensor(delta, device=device, dtype=dtype)
    eps_t = torch.tensor(eps, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    loss_hist = []

    for ep in range(1, epochs + 1):
        idx_list = torch.randint(1, T, (K_time,), device="cpu").tolist()

        gradA = torch.zeros_like(A)
        gradx = torch.zeros_like(x_star)
        loss_acc = torch.zeros((), device=device, dtype=dtype)
        A_eff = apply_mask(A)

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
            rnorm = torch.linalg.norm(r_mat, dim=0)
            alpha = torch.minimum(one, delta_t / (rnorm + eps_t))
            g_mat = -r_mat
            D_alpha = D * alpha.unsqueeze(0)
            E_total = t * (D_alpha @ g_mat.T)
            L_total = expm_frechet_from_cache(cache, E_total)
            gradA = gradA + (-(L_total.T))

            Bmat = I - R
            r_weighted = r_mat * alpha.unsqueeze(0)
            gradx = gradx + (Bmat.T @ r_weighted).sum(dim=1)

            huber_each = torch.where(
                rnorm <= delta_t,
                0.5 * rnorm**2,
                delta_t * (rnorm - 0.5 * delta_t),
            )
            loss_acc = loss_acc + huber_each.sum() / G

        denom = max(1, (B * K_time))
        gradA = gradA / (denom * G)
        gradx = gradx / (denom * G)

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

        if do_print and (print_every is not None) and (ep % print_every == 0 or ep == 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(
                f"epoch {ep:04d} | huber loss {(loss_acc/denom).item():.4e} | "
                f"||A||_F {torch.linalg.norm(A).item():.3e} | "
                f"||x*|| {torch.linalg.norm(x_star).item():.3e}"
            )

        loss_hist.append(float((loss_acc / denom).item()))

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_sec = time.time() - t0

    A_eff = apply_mask(A)
    E_all = torch.matrix_exp(A_eff.unsqueeze(0) * t_span.view(T, 1, 1))
    Y_pred = torch.einsum("tij,bj->bti", E_all, (X0 - x_star)) + x_star

    return {
        "name": "our_model",
        "display_name": "OurModel",
        "time_sec": total_sec,
        "final_loss": loss_hist[-1] if loss_hist else None,
        "loss_history": loss_hist,
        "A_hat": A,
        "x_star_hat": x_star,
        "Y_pred": Y_pred,
    }



def compute_our_model_epoch0_loss(X0, Y_obs, t_span, Mask, init_A):
    device = Y_obs.device
    dtype = Y_obs.dtype
    Mask_f = Mask.to(device=device, dtype=dtype)
    with torch.no_grad():
        x_star0 = Y_obs[:, -10:, :].mean(dim=(0, 1)).detach().clone().to(device=device, dtype=dtype)
        A0 = init_A * Mask_f
        E0 = torch.matrix_exp(A0.unsqueeze(0) * t_span.view(t_span.shape[0], 1, 1))
        Y0 = torch.einsum("tij,bj->bti", E0, (X0 - x_star0)) + x_star0
        loss0 = F.mse_loss(Y0, Y_obs).item()
    return loss0, Y0
