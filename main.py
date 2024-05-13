from utils.path import add_submodules_to_path, out_dir

add_submodules_to_path()

from conditional.fpssmc import FPSSMC
from conditional.replacement import ReplacementMethod
from conditional.tds import TDS
import hydra
import logging
from model.genie import GenieAdapter
from omegaconf import DictConfig, OmegaConf
import os
import sys
import time
import torch
import traceback
from types import TracebackType
from typing import Callable
from utils.pdb import pdb_to_c_alpha_backbone, c_alpha_backbone_to_pdb, get_motif_mask
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
def sample_given_motif(cfg):
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

    _motif = pdb_to_c_alpha_backbone(cfg.experiment.motif).to(device)
    motif, motif_mask = get_motif_mask(
        _motif,
        cfg.experiment.sample_length,
        model.max_n_residues,
        cfg.experiment.motif_contig_region,
    )

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
            os.path.join(out, "scaffolds", f"scaffold-{i}.pdb"),
        )

    c_alpha_backbone_to_pdb(
        motif[0][motif_mask[0] == 1],
        os.path.join(out, f"motif.pdb"),
    )

    if cfg.experiment.keep_coords_trace:
        os.makedirs(os.path.join(out, "traces"))
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans[mask[0] == 1].detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(out, "traces", f"trace-{i}.pt"))


@experiment_job
def sample_unconditional(cfg):
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
            os.path.join(out, "samples", f"sample-{i}.pdb"),
        )

    if cfg.experiment.keep_coords_trace:
        os.makedirs(os.path.join(out, "traces"))
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans.detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(out, "traces", f"trace-{i}.pt"))


@experiment_job
def sample_given_symmetry(cfg):
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
            os.path.join(out, "scaffolds", f"scaffold-{i}.pdb"),
        )

    if cfg.experiment.keep_coords_trace:
        os.makedirs(os.path.join(out, "traces"))
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans[mask[0] == 1].detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(out, "traces", f"trace-{i}.pt"))


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    experiment_name = cfg.experiment["name"]
    NO_EXPERIMENT_SET = "none"

    if experiment_name == NO_EXPERIMENT_SET:
        print("Run with the --help flag to see options for experiments.")
        return

    if experiment_name == "sample_unconditional":
        sample_unconditional(cfg)
        return

    if experiment_name == "sample_given_motif":
        sample_given_motif(cfg)
        return

    if experiment_name == "sample_given_symmetry":
        sample_given_symmetry(cfg)
        return


def log_exception(_type: type, value: BaseException, tb: TracebackType) -> None:
    if issubclass(_type, KeyboardInterrupt):
        sys.__excepthook__(_type, value, tb)
        return
    logger.exception("".join(traceback.format_exception(_type, value, tb)))


sys.excepthook = log_exception

if __name__ == "__main__":
    main()
