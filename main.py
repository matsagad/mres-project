from utils.path import add_submodules_to_path, out_dir

add_submodules_to_path()

from model.genie import GenieAdapter
from conditional.replacement import ReplacementMethod
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import os
import sys
import time
import torch
from typing import Callable
from utils.pdb import pdb_to_c_alpha_backbone, c_alpha_backbone_to_pdb
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


@experiment_job
def sample_conditional(cfg):
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

    motif = pdb_to_c_alpha_backbone(cfg.experiment.motif).to(device)
    motif_mask = torch.zeros((1, model.max_n_residues), device=device)
    # Currently we assume motif is contiguous and placed at the start
    motif_mask[:, : motif.shape[0]] = 1

    mask = torch.zeros((cfg.experiment.n_samples, model.max_n_residues), device=device)
    mask[:, : cfg.experiment.sample_length] = 1

    cond_cfg = cfg.experiment.conditional_method
    method = cond_cfg.name

    setup = None
    if method == "replacement" or method == "smcdiff":
        resampling_method = None
        if method == "smcdiff":
            if cond_cfg.resampling_method not in RESAMPLING_METHOD:
                logger.error(
                    f'Invalid resampling method "{cond_cfg.resampling_method}". '
                    f"Choose from: {', '.join(RESAMPLING_METHOD.keys())}."
                )
                sys.exit(1)
            resampling_method = RESAMPLING_METHOD[cond_cfg.resampling_method]

        setup = ReplacementMethod(model).with_config(
            noisy_motif=cond_cfg.noisy_motif,
            particle_filter=(method == "smcdiff"),
            replacement_weight=float(cond_cfg.replacement_weight),
            resample_indices=resampling_method,
        )

    samples = setup.sample_given_motif(mask, motif, motif_mask)

    out = out_dir()
    os.makedirs(os.path.join(out, "scaffolds"))
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans.detach().cpu(),
            os.path.join(out, "scaffolds", f"scaffold-{i}.pdb"),
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


@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    experiment_name = cfg.experiment["name"]
    NO_EXPERIMENT_SET = "none"

    if experiment_name == NO_EXPERIMENT_SET:
        print("Run with the --help flag to see options for experiments.")
        return

    if experiment_name == "sample_conditional":
        sample_conditional(cfg)
        return

    if experiment_name == "sample_unconditional":
        sample_unconditional(cfg)
        return


if __name__ == "__main__":
    main()
