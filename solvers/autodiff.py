import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class LTIExact(nn.Module):
    def __init__(self, d, mask):
        super().__init__()
        self.d = d
        self.register_buffer("mask", mask.bool())
        self.W = nn.Parameter(torch.zeros(d, d))
        self.x_star = nn.Parameter(torch.zeros(d))

    @property
    def A_eff(self):
        return self.W * self.mask

    def predict_traj(self, X0, t_span):
        A = self.A_eff
        T = t_span.numel()
        E = torch.matrix_exp(A.unsqueeze(0) * t_span.view(T, 1, 1))
        return torch.einsum("bid,tdk->btk", (X0 - self.x_star)[:, None, :], E.transpose(-1, -2)) + self.x_star

    def forward(self, X0, t_span):
        return self.predict_traj(X0, t_span)


def _predict_from_params(W, x_star, mask, X0, t_span):
    A_eff = W * mask.to(device=W.device, dtype=W.dtype)
    T = t_span.numel()
    E = torch.matrix_exp(A_eff.unsqueeze(0) * t_span.view(T, 1, 1))
    return torch.einsum("bid,tdk->btk", (X0 - x_star)[:, None, :], E.transpose(-1, -2)) + x_star


def run_autodiff_LTI(
    NumAllGene,
    Mask,
    X0,
    t_span,
    Y_obs,
    device=None,
    epochs=5000,
    reg_lambda=1e-4,
    time_it=True,
    print_every=1000,
    verbose=True,
    init_W=None,
    K_time=None,
    lr_W=3e-3,
    lr_x=3e-3,
):
    """
    Torch-optimizer-free autodiff training.
    This avoids torch.optim.Adam, which can break on some local torch installs
    that raise: AttributeError: module 'torch' has no attribute '_utils'.
    """
    if device is None:
        device = X0.device

    dtype = Y_obs.dtype
    mask_f = Mask.to(device=device, dtype=dtype)

    if init_W is None:
        W = (0.01 * torch.randn(NumAllGene, NumAllGene, device=device, dtype=dtype)) * mask_f
    else:
        W = init_W.to(device=device, dtype=dtype).clone()
    W = (W * mask_f).detach().requires_grad_(True)

    x_star = Y_obs[:, -10:, :].mean(dim=(0, 1)).to(device=device, dtype=dtype).detach().clone().requires_grad_(True)

    beta1, beta2 = 0.9, 0.999
    adam_eps = 1e-8
    mW = torch.zeros_like(W)
    vW = torch.zeros_like(W)
    mx = torch.zeros_like(x_star)
    vx = torch.zeros_like(x_star)

    total_sec = None
    if time_it:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

    loss_history = []
    T = Y_obs.shape[1]

    for epoch in range(1, epochs + 1):
        if W.grad is not None:
            W.grad = None
        if x_star.grad is not None:
            x_star.grad = None

        if (K_time is None) or (K_time >= T):
            Y_pred = _predict_from_params(W, x_star, mask_f, X0, t_span)
            loss = F.mse_loss(Y_pred, Y_obs)
        else:
            idx = torch.randint(1, T, (K_time,), device=device)
            t_sub = t_span.index_select(0, idx)
            Y_pred = _predict_from_params(W, x_star, mask_f, X0, t_sub)
            loss = F.mse_loss(Y_pred, Y_obs.index_select(1, idx))

        loss = loss + reg_lambda * (W * mask_f).pow(2).sum()
        loss.backward()

        with torch.no_grad():
            gW = W.grad
            gx = x_star.grad

            mW.mul_(beta1).add_(gW, alpha=1 - beta1)
            vW.mul_(beta2).addcmul_(gW, gW, value=1 - beta2)
            mW_hat = mW / (1 - beta1**epoch)
            vW_hat = vW / (1 - beta2**epoch)
            W.addcdiv_(mW_hat, torch.sqrt(vW_hat) + adam_eps, value=-lr_W)
            W.mul_(mask_f)

            mx.mul_(beta1).add_(gx, alpha=1 - beta1)
            vx.mul_(beta2).addcmul_(gx, gx, value=1 - beta2)
            mx_hat = mx / (1 - beta1**epoch)
            vx_hat = vx / (1 - beta2**epoch)
            x_star.addcdiv_(mx_hat, torch.sqrt(vx_hat) + adam_eps, value=-lr_x)

        loss_history.append(float(loss.item()))

        if verbose and print_every is not None and ((epoch - 1) % print_every == 0):
            print(f"epoch {epoch-1:04d} | loss {loss.item():.4e} | ||x*|| {x_star.norm().item():.3e}")

    if time_it:
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_sec = time.perf_counter() - t_start

    with torch.no_grad():
        Y_pred_full = _predict_from_params(W, x_star, mask_f, X0, t_span)
        model = LTIExact(NumAllGene, Mask).to(device=device, dtype=dtype)
        model.W.copy_(W.detach())
        model.x_star.copy_(x_star.detach())

    return {
        "name": "autodiff",
        "display_name": "Autodiff",
        "time_sec": total_sec,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "model": model,
        "Y_pred": Y_pred_full,
    }
