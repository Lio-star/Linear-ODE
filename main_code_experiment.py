from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from model_runner import MODEL_REGISTRY, build_shared_init_A, fit_models, normalize_models
from utils.data_generation import build_mask_and_generate_data
from utils.plotting import plot_seed_losses, plot_seed_trajectories
from utils.summary import (
    build_summary_table,
    print_final_summary_banner,
    save_summary_tables,
)


class MainCodeExperiment:
    

    def __init__(self, env=2, T=100, output_dir="outputs", default_dtype=torch.float32):
        self.env = env
        self.T = T
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"

        self.device = torch.device(
            "cuda" if (env == 2 and torch.cuda.is_available()) else "cpu"
        )
        self.default_dtype = default_dtype

        torch.set_default_dtype(default_dtype)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def run(
        self,
        configs,
        data_seeds,
        models,
        epochs=5000,
        checkpoints=None,
        init_seed=0,
        k_time=1,
        plot_each_seed=True,
        save_figures=False,
        display_figures=True,
        save_tables=False,
        print_checkpoint_losses=True,
        verbose=False,
        print_every=None,
    ):
        models = normalize_models(models)
        checkpoints = checkpoints or [0, 1000, 2000, 3000, 4000, 5000]

        if save_figures:
            self.figures_dir.mkdir(parents=True, exist_ok=True)

        results = []
        run_details = []

        for cfg in configs:
            NumAllGene = cfg["NumAllGene"]
            NumTF = cfg["NumTF"]
            B = cfg["B"]

            for run_idx, data_seed in enumerate(data_seeds, start=1):
                print("" + "=" * 70)
                print(
                    f"CONFIG: G={NumAllGene}, TF={NumTF}, B={B} | "
                    f"DATA RUN {run_idx}/{len(data_seeds)} | data_seed={data_seed}"
                )
                print("=" * 70)

                self._set_seed(data_seed)

                A_true, Mask, X0, t_span, x_star_true, Y_clean, Y_obs = (
                    build_mask_and_generate_data(
                        NumAllGene=NumAllGene,
                        NumTF=NumTF,
                        B=B,
                        T=self.T,
                        seed=data_seed,
                        device=self.device,
                        dtype=self.default_dtype,
                    )
                )

                G = Y_obs.shape[2]
                dtype = Y_obs.dtype
                init_A = build_shared_init_A(
                    G=G,
                    Mask=Mask,
                    device=self.device,
                    dtype=dtype,
                    init_seed=init_seed,
                )

                outputs, epoch0 = fit_models(
                    models=models,
                    NumAllGene=NumAllGene,
                    Mask=Mask,
                    X0=X0,
                    t_span=t_span,
                    Y_obs=Y_obs,
                    init_A=init_A,
                    epochs=epochs,
                    K_time=k_time,
                    device=self.device,
                    init_seed=init_seed,
                    print_every=print_every,
                    verbose=verbose,
                )

                if print_checkpoint_losses:
                    print("Checkpoint losses (after k epochs):")
                    for k in checkpoints:
                        parts = []
                        for model_key in models:
                            display_name = MODEL_REGISTRY[model_key]["display_name"]
                            if k == 0:
                                loss_value = epoch0[model_key]
                            else:
                                hist = outputs[model_key]["loss_history"]
                                idx = min(max(k - 1, 0), len(hist) - 1)
                                loss_value = hist[idx]
                            parts.append(f"{display_name}: {loss_value:.6e}")
                        print(f" epoch {k:4d} | " + " | ".join(parts))

                title = (
                    f"G={NumAllGene}, TF={NumTF}, B={B} | "
                    f"data_run={run_idx} (seed={data_seed})"
                )
                predictions = {
                    outputs[m]["display_name"]: outputs[m]["Y_pred"] for m in models
                }
                loss_histories = {
                    outputs[m]["display_name"]: outputs[m]["loss_history"] for m in models
                }

                if plot_each_seed:
                    traj_path = None
                    
                    if save_figures:
                        stem = (
                            f"G{NumAllGene}_TF{NumTF}_B{B}_"
                            f"run{run_idx:02d}_seed{data_seed}"
                        )
                        traj_path = self.figures_dir / f"{stem}_trajectory.png"
                        
                    plot_seed_trajectories(
                        t_span=t_span,
                        Y_obs=Y_obs,
                        Y_true=Y_clean,
                        predictions=predictions,
                        title_prefix=title,
                        save_path=traj_path,
                        show=display_figures,
                    )

                    

                row = {
                    "NumAllGene": NumAllGene,
                    "NumTF": NumTF,
                    "B": B,
                    "DataRun": run_idx,
                    "DataSeed": data_seed,
                    "K_time": k_time,
                }

                for model_key in models:
                    row[MODEL_REGISTRY[model_key]["time_col"]] = outputs[model_key]["time_sec"]
                    row[MODEL_REGISTRY[model_key]["loss_col"]] = outputs[model_key]["final_loss"]

                results.append(row)

                run_details.append(
                    {
                        "config": cfg,
                        "data_seed": data_seed,
                        "epoch0": epoch0,
                        "outputs": outputs,
                        "A_true": A_true,
                        "Mask": Mask,
                        "X0": X0,
                        "t_span": t_span,
                        "x_star_true": x_star_true,
                        "Y_clean": Y_clean,
                        "Y_obs": Y_obs,
                    }
                )

        df_final = build_summary_table(results, models)
        df_final = df_final.round(6)

        print_final_summary_banner()

        #if save_tables:
            #save_summary_tables(df_final, self.output_dir)

        metadata = {
            "results": results,
            "run_details": run_details,
            "figures_dir": self.figures_dir if save_figures else None,
        }
        return df_final, metadata
