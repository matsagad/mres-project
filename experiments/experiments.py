from conditional import CONDITIONAL_METHOD_REGISTRY
from experiments import register_experiment
import logging
from model.genie import GenieAdapter
import numpy as np
from omegaconf import DictConfig
import omegaconf
import os
import subprocess
import torch
from utils.path import out_dir
from utils.pdb import (
    pdb_to_atom_backbone,
    pdb_to_c_alpha_backbone,
    atom_backbone_to_pdb,
    c_alpha_backbone_to_pdb,
    get_motif_mask,
)

logger = logging.getLogger(__name__)


@register_experiment("sample_given_motif")
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

    motif_cfg = cfg.experiment.motif
    motif_backbones = pdb_to_atom_backbone(motif_cfg.path)
    motif_mask, mask, masked_backbones = get_motif_mask(
        motif_backbones,
        model.max_n_residues,
        motif_cfg.contig_region,
        return_masked_backbones=True,
    )
    motif = masked_backbones["CA"].to(device)
    motif_mask = motif_mask.to(device)
    mask = torch.tile(mask, (cfg.experiment.n_samples, 1)).to(device)

    cond_cfg = cfg.experiment.conditional_method
    method = cond_cfg.method

    if method not in CONDITIONAL_METHOD_REGISTRY:
        raise Exception(
            f"Invalid method supplied: {method}. Valid options: {', '.join(CONDITIONAL_METHOD_REGISTRY.keys())}."
        )
    conditional_wrapper, config_resolver = CONDITIONAL_METHOD_REGISTRY[method]

    setup = conditional_wrapper(model).with_config(**config_resolver(cond_cfg))
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
    with open(os.path.join(out, "motif_cfg.yaml"), "w") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in dict(motif_cfg).items()))

    if cfg.experiment.keep_coords_trace:
        os.makedirs(os.path.join(out, "traces"))
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans[:, mask[0] == 1].detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(out, "traces", f"trace_{i}.pt"))


@register_experiment("sample_unconditional")
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

    # WLOG just choose any method and call on their unconditional sample method.
    conditional_method, _ = CONDITIONAL_METHOD_REGISTRY["smcdiff"]
    setup = conditional_method(model)
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


@register_experiment("sample_given_symmetry")
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

    conditional_wrapper, config_resolver = CONDITIONAL_METHOD_REGISTRY[method]
    if not conditional_wrapper.supports_condition_on_symmetry:
        raise Exception(
            f"Conditional method {method} does not support conditioning on symmetry."
        )
    setup = conditional_wrapper(model).with_config(**config_resolver(cond_cfg))

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


@register_experiment("evaluate_samples")
def evaluate_samples(cfg: DictConfig) -> None:
    path_to_samples = cfg.experiment.path_to_samples
    path_to_motif = cfg.experiment.path_to_motif
    path_to_motif_cfg = cfg.experiment.path_to_motif_cfg

    out = out_dir()
    n_samples = 0

    # Populate coords folder with CA atom coordinates of each sample
    path_to_coords = os.path.join(out, "coords")
    os.makedirs(path_to_coords)
    sample_lengths = []
    with os.scandir(path_to_samples) as files:
        for file in files:
            if file.is_file() and file.name.endswith(".pdb"):
                n_samples += 1
                c_alpha_coords = pdb_to_c_alpha_backbone(
                    os.path.join(path_to_samples, file.name)
                ).numpy()
                out_name = f"{file.name.split('.')[0]}.npy"
                np.savetxt(
                    os.path.join(out, "coords", out_name),
                    c_alpha_coords,
                    fmt="%.3f",
                    delimiter=",",
                )
                sample_lengths.append(len(c_alpha_coords))

    # Load motif config file
    motif_cfg = omegaconf.OmegaConf.load(path_to_motif_cfg)
    motif_backbones = pdb_to_atom_backbone(path_to_motif)
    motif_mask, _ = get_motif_mask(
        motif_backbones,
        max(sample_lengths),
        motif_cfg.contig_region,
        return_masked_backbones=False,
    )

    # Populate motif_masks folder with bitmasks for each sample
    path_to_masks = os.path.join(out, "motif_masks")
    os.makedirs(path_to_masks)
    for i in range(n_samples):
        np.savetxt(
            os.path.join(path_to_masks, f"scaffold_{i}.npy"),
            # Motif mask must be same size as sample lengths
            motif_mask[0, : sample_lengths[i]].numpy(),
            fmt="%.3f",
            delimiter=",",
        )
    np.savetxt(
        os.path.join(path_to_masks, "motif.npy"),
        motif_backbones["CA"],
        fmt="%.3f",
        delimiter=",",
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