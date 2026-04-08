import numpy as np
import torch


def sample_A(
    G=100,
    Mask=None,
    density=0.08,
    act_frac=0.6,
    weight_scale=0.1,
    deg_minmax=(0.2, 0.8),
    stabilize="diag_dom",
    shift_margin=0.05,
    seed=20,
    device=None,
    dtype=torch.float32,
):
    if device is None:
        if Mask is not None:
            device = Mask.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    G = int(G)
    mask = (torch.rand(G, G, device=device) < density)
    mask.fill_diagonal_(False)

    one = torch.tensor(1.0, device=device, dtype=dtype)
    neg_one = torch.tensor(-1.0, device=device, dtype=dtype)

    signs = torch.where(torch.rand(G, G, device=device) < act_frac, one, neg_one)
    W = torch.randn(G, G, device=device, dtype=dtype) * weight_scale
    W = W * signs * mask
    W.fill_diagonal_(0.0)

    deg = torch.empty(G, device=device, dtype=dtype).uniform_(*deg_minmax)
    A = (W.clone() * Mask).to(dtype=dtype)
    A.diagonal().copy_(-deg)

    if stabilize == "diag_dom":
        row_sum = torch.sum(torch.abs(A), dim=1) - torch.abs(torch.diag(A))
        extra = torch.clamp(row_sum - (-torch.diag(A)) + 1e-3, min=0.0)
        A = A.clone()
        A[range(G), range(G)] -= extra
    elif stabilize == "shift":
        smax = torch.linalg.svdvals(W).max()
        alpha = float(smax) + shift_margin + float(deg.min())
        A = W - alpha * torch.eye(G, device=device, dtype=dtype)
    else:
        raise ValueError("stabilize must be 'diag_dom' or 'shift'")

    return A



def build_mask_and_generate_data(
    NumAllGene=50,
    NumTF=20,
    B=2,
    T=100,
    t0=0.0,
    t1=1.0,
    noise_std=0.01,
    density=0.5,
    act_frac=0.5,
    weight_scale=1.5,
    deg_minmax=(0.3, 0.5),
    stabilize="diag_dom",
    seed=4,
    device=None,
    dtype=None,
):
    NumTG = NumAllGene - NumTF

    A11 = np.eye(NumTG)
    A12 = np.ones((NumTG, NumTF))
    A21 = np.zeros((NumTF, NumTG))
    A22 = np.ones((NumTF, NumTF))
    BLK = np.block([[A11, A12], [A21, A22]])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Mask = torch.tensor(BLK, dtype=torch.bool, device=device)

    if dtype is None:
        dtype = torch.float32

    A_true = sample_A(
        NumAllGene,
        Mask,
        density=density,
        act_frac=act_frac,
        weight_scale=weight_scale,
        deg_minmax=deg_minmax,
        stabilize=stabilize,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    device = A_true.device
    dtype = A_true.dtype
    G = A_true.shape[0]

    t_span = torch.linspace(t0, t1, T, device=device, dtype=dtype)
    X0 = torch.randn(B, G, device=device, dtype=dtype)
    E = torch.matrix_exp(A_true.unsqueeze(0) * t_span.view(T, 1, 1))

    x_star_true = torch.rand(G, device=device, dtype=dtype)
    Y_clean = torch.einsum("bg,tkg->btk", (X0 - x_star_true), E.transpose(-1, -2)) + x_star_true
    Y_obs = Y_clean + noise_std * torch.randn_like(Y_clean)

    return A_true, Mask, X0, t_span, x_star_true, Y_clean, Y_obs
