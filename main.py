from utils.path import add_submodules_to_path, out_dir

add_submodules_to_path()

from conditional.fpssmc import FPSSMC
from conditional.replacement import ReplacementMethod
from conditional.tds import TDS
import hydra
import logging
from model.genie import GenieAdapter
import numpy as np
from omegaconf import DictConfig, OmegaConf
import omegaconf
import os
import subprocess
import sys
import time
import torch
import traceback
from types import TracebackType
from typing import Callable
from utils.pdb import (
    pdb_to_atom_backbone,
    pdb_to_c_alpha_backbone,
    atom_backbone_to_pdb,
    c_alpha_backbone_to_pdb,
    get_motif_mask,
)
from utils.resampling import RESAMPLING_METHOD

logger = logging.getLogger(__name__)


def experiment_job(fn: Callable) -> Callable:
    def wrapper(cfg: DictConfig):
        experiment_name = cfg.experiment["name"]

        t_start = time.time()
        logger.info(f"Started {experiment_name} experiment.")

        fn(cfg)

        t_elapsed_str = time.strftime(f"%H:%M:%S", time.gmtime(time.time() - t_start))
        logger.info(f"Finished {experiment_name} experiment in {t_elapsed_str}.")

    return wrapper


def get_resampling_method(method: str) -> Callable:
    if method not in RESAMPLING_METHOD:
        logger.error(
            f'Invalid resampling method "{method}". '
            f"Choose from: {', '.join(RESAMPLING_METHOD.keys())}."
        )
        sys.exit(1)
    return RESAMPLING_METHOD[method]


@experiment_job
def sample_given_motif(cfg: DictConfig) -> None:
    device = torch.device(cfg.model.device)

    model = (
        GenieAdapter.from_weights_and_config(cfg.model.weights, cfg.model.config)
        .with_batch_size(cfg.model.batch_size)
        .with_noise_scale(cfg.model.noise_scale)
        .to(device)
    )

    # Both model.max_n_residues and model.n_timesteps are already set
    # according to Genie's custom config file above. Although, another option
    # is to set them with values from cfg in case we use other models.

    motif_backbones = pdb_to_atom_backbone(cfg.experiment.motif)
    masked_backbones, motif_mask = get_motif_mask(
        motif_backbones,
        cfg.experiment.sample_length,
        model.max_n_residues,
        cfg.experiment.motif_contig_region,
        mask_backbones=True,
    )
    motif = masked_backbones["CA"].to(device)
    motif_mask = motif_mask.to(device)

    mask = torch.zeros((cfg.experiment.n_samples, model.max_n_residues), device=device)
    mask[:, : cfg.experiment.sample_length] = 1

    cond_cfg = cfg.experiment.conditional_method
    method = cond_cfg.name

    setup = None
    resampling_method = (
        None
        if method == "replacement"
        else get_resampling_method(cond_cfg.resampling_method)
    )

    if method == "replacement" or method == "smcdiff":
        setup = ReplacementMethod(model).with_config(
            noisy_motif=cond_cfg.noisy_motif,
            particle_filter=(method == "smcdiff"),
            replacement_weight=float(cond_cfg.replacement_weight),
            resample_indices=resampling_method,
        )

    elif method == "tds":
        setup = TDS(model).with_config(
            resample_indices=resampling_method,
        )

    elif method == "fpssmc":
        setup = FPSSMC(model).with_config(
            noisy_motif=cond_cfg.noisy_motif,
            particle_filter=cond_cfg.particle_filter,
            resample_indices=resampling_method,
            sigma=float(cond_cfg.sigma),
        )

    samples = setup.sample_given_motif(mask, motif, motif_mask)

    out = out_dir()
    os.makedirs(os.path.join(out, "scaffolds"))
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(out, "scaffolds", f"scaffold_{i}.pdb"),
        )

    atom_backbone_to_pdb(
        {
            atom: backbone[:, (motif_mask[0] == 1).cpu()]
            for atom, backbone in masked_backbones.items()
        },
        os.path.join(out, "motif.pdb"),
    )

    motif_cfg = {
        "sample_length": cfg.experiment.sample_length,
        "motif_contig_region": cfg.experiment.motif_contig_region,
    }
    with open(os.path.join(out, "motif_cfg.yaml"), "w") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in motif_cfg.items()))

    if cfg.experiment.keep_coords_trace:
        os.makedirs(os.path.join(out, "traces"))
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans[mask[0] == 1].detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(out, "traces", f"trace_{i}.pt"))


@experiment_job
def sample_unconditional(cfg: DictConfig) -> None:
    device = torch.device(cfg.model.device)

    model = (
        GenieAdapter.from_weights_and_config(cfg.model.weights, cfg.model.config)
        .with_batch_size(cfg.model.batch_size)
        .with_noise_scale(cfg.model.noise_scale)
        .to(device)
    )

    mask = torch.zeros((cfg.experiment.n_samples, model.max_n_residues), device=device)
    mask[:, : cfg.experiment.sample_length] = 1

    setup = ReplacementMethod(model)
    samples = setup.sample(mask)

    out = out_dir()
    os.makedirs(os.path.join(out, "samples"))
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans.detach().cpu(),
            os.path.join(out, "samples", f"sample_{i}.pdb"),
        )

    if cfg.experiment.keep_coords_trace:
        os.makedirs(os.path.join(out, "traces"))
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans.detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(out, "traces", f"trace_{i}.pt"))


@experiment_job
def sample_given_symmetry(cfg: DictConfig) -> None:
    device = torch.device(cfg.model.device)

    model = (
        GenieAdapter.from_weights_and_config(cfg.model.weights, cfg.model.config)
        .with_batch_size(cfg.model.batch_size)
        .with_noise_scale(cfg.model.noise_scale)
        .to(device)
    )

    mask = torch.zeros((cfg.experiment.n_samples, model.max_n_residues), device=device)
    mask[:, : cfg.experiment.sample_length] = 1

    cond_cfg = cfg.experiment.conditional_method
    method = cond_cfg.name

    setup = None
    resampling_method = get_resampling_method(cond_cfg.resampling_method)

    if method == "fpssmc":
        setup = FPSSMC(model).with_config(
            noisy_motif=cond_cfg.noisy_motif,
            particle_filter=cond_cfg.particle_filter,
            resample_indices=resampling_method,
            sigma=float(cond_cfg.sigma),
        )

    samples = setup.sample_given_symmetry(mask, cfg.experiment.symmetry)

    out = out_dir()
    os.makedirs(os.path.join(out, "scaffolds"))
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(out, "scaffolds", f"scaffold_{i}.pdb"),
        )

    if cfg.experiment.keep_coords_trace:
        os.makedirs(os.path.join(out, "traces"))
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans[mask[0] == 1].detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(out, "traces", f"trace_{i}.pt"))


@experiment_job
def evaluate_samples(cfg: DictConfig) -> None:
    path_to_samples = cfg.experiment.path_to_samples
    path_to_motif = cfg.experiment.path_to_motif
    path_to_motif_cfg = cfg.experiment.path_to_motif_cfg

    out = out_dir()
    n_samples = 0

    # Populate coords folder with CA atom coordinates of each sample
    path_to_coords = os.path.join(out, "coords")
    os.makedirs(path_to_coords)
    with os.scandir(path_to_samples) as files:
        for file in files:
            if file.is_file() and file.name.endswith(".pdb"):
                n_samples += 1
                c_alpha_coords = pdb_to_c_alpha_backbone(
                    os.path.join(path_to_samples, file.name)
                ).numpy()
                out_name = f"{file.name.split('.')[0]}.npy"
                np.savetxt(
                    os.path.join(out, "coords", out_name), c_alpha_coords, delimiter=","
                )

    # Load motif config file
    motif_cfg = omegaconf.OmegaConf.load(path_to_motif_cfg)
    motif_backbones = pdb_to_atom_backbone(path_to_motif)
    motif_mask = get_motif_mask(
        motif_backbones,
        motif_cfg.sample_length,
        motif_cfg.sample_length,
        motif_cfg.motif_contig_region,
        mask_backbones=False,
    )

    # Populate motif_masks folder with bitmasks for each sample
    path_to_masks = os.path.join(out, "motif_masks")
    os.makedirs(path_to_masks)
    for i in range(n_samples):
        np.savetxt(
            os.path.join(path_to_masks, f"scaffold_{i}.npy"),
            motif_mask[0].numpy(),
        )

    eval_out = os.path.join(out, "evaluation")
    eval_cmd_args = [
        "python3",
        "evaluations/pipeline/evaluate.py",
        f"--input_dir={out}",  # Must contain "coords" and "motif_masks" subfolders
        f"--output_dir={eval_out}",
    ]
    if path_to_motif:
        eval_cmd_args.append(f"--motif_filepath={path_to_motif}")
    if not os.path.isdir("submodules/genie/packages"):
        raise Exception(
            "Genie packages folder missing: have you run `submodules/genie/scripts/setup_evaluation_pipeline.sh`?"
        )
    eval = subprocess.Popen(
        eval_cmd_args,
        cwd=os.path.join(os.getcwd(), "submodules/genie"),
    )
    eval.wait()


def placeholder_job(*args) -> None:
    print("Run with the --help flag to see options for experiments.")


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    experiment_name = cfg.experiment["name"]
    NO_EXPERIMENT_SET = "none"

    experiment_jobs = {
        NO_EXPERIMENT_SET: placeholder_job,
        "sample_unconditional": sample_unconditional,
        "sample_given_motif": sample_given_motif,
        "sample_given_symmetry": sample_given_symmetry,
        "evaluate_samples": evaluate_samples,
    }

    experiment_jobs.get(experiment_name, placeholder_job)(cfg)


def log_exception(_type: type, value: BaseException, tb: TracebackType) -> None:
    if issubclass(_type, KeyboardInterrupt):
        sys.__excepthook__(_type, value, tb)
        return
    logger.exception("".join(traceback.format_exception(_type, value, tb)))


sys.excepthook = log_exception

if __name__ == "__main__":
    main()
