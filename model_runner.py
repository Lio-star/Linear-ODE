import torch
import torch.nn.functional as F

from solvers.autodiff import LTIExact, run_autodiff_LTI
from solvers.our_model import run_ourmethod_learn_A_xstar, compute_our_model_epoch0_loss


MODEL_REGISTRY = {
    "autodiff": {"display_name": "Autodiff", "time_col": "Autodiff_Time", "loss_col": "Autodiff_Loss"},
    "our_model": {"display_name": "OurModel", "time_col": "OurModel_Time", "loss_col": "OurModel_Loss"},
}

ALIASES = {
    "autodiff": "autodiff",
    "our_model": "our_model",
    "ourmodel": "our_model",
    "our-method": "our_model",
    "ourmethod": "our_model",
}



def normalize_models(models):
    normalized = []
    for model in models:
        key = str(model).strip().lower()
        if key not in ALIASES:
            raise ValueError(f"Unsupported model '{model}'. Supported: {list(ALIASES)}")
        canonical = ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized



def build_shared_init_A(G, Mask, device, dtype, init_seed):
    Mask_f = Mask.to(device=device, dtype=dtype)
    torch.manual_seed(init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(init_seed)
    return (0.01 * torch.randn(G, G, device=device, dtype=dtype)) * Mask_f



def compute_autodiff_epoch0_loss(NumAllGene, Mask, X0, t_span, Y_obs, init_A, reg_lambda=1e-6):
    dtype = Y_obs.dtype
    device = Y_obs.device
    with torch.no_grad():
        model0 = LTIExact(NumAllGene, Mask).to(device=device, dtype=dtype)
        model0.W.copy_(init_A.to(device=device, dtype=model0.W.dtype))
        Y0 = model0(X0, t_span)
        loss0 = (F.mse_loss(Y0, Y_obs) + reg_lambda * (model0.W * model0.mask).pow(2).sum()).item()
    return loss0, Y0



def fit_models(
    models,
    NumAllGene,
    Mask,
    X0,
    t_span,
    Y_obs,
    init_A,
    epochs,
    K_time,
    device,
    init_seed,
    print_every=None,
    verbose=False,
):
    models = normalize_models(models)
    outputs = {}
    epoch0 = {}

    if "autodiff" in models:
        loss0, _ = compute_autodiff_epoch0_loss(NumAllGene, Mask, X0, t_span, Y_obs, init_A)
        epoch0["autodiff"] = loss0
        outputs["autodiff"] = run_autodiff_LTI(
            NumAllGene=NumAllGene,
            Mask=Mask,
            X0=X0,
            t_span=t_span,
            Y_obs=Y_obs,
            device=device,
            epochs=epochs,
            reg_lambda=1e-6,
            time_it=True,
            print_every=print_every,
            verbose=verbose,
            init_W=init_A,
            K_time=K_time,
            lr_W=3e-3,
            lr_x=3e-3,
        )

    if "our_model" in models:
        loss0, _ = compute_our_model_epoch0_loss(X0, Y_obs, t_span, Mask, init_A)
        epoch0["our_model"] = loss0
        outputs["our_model"] = run_ourmethod_learn_A_xstar(
            X0=X0,
            Y_obs=Y_obs,
            t_span=t_span,
            Mask=Mask,
            epochs=epochs,
            lr_A=3e-3,
            lr_xs=3e-3,
            K_time=K_time,
            delta=0.07,
            eps=1e-12,
            print_every=print_every,
            seed=init_seed,
            do_print=verbose,
            init_A=init_A,
        )

    return outputs, epoch0
