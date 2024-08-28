from conditional import CONDITIONAL_METHOD_REGISTRY
from experiments import register_experiment
import logging
from model import DIFFUSION_MODEL_REGISTRY
from model.diffusion import FrameDiffusionModel
import omegaconf
from omegaconf import DictConfig
import os
from pathlib import Path
import subprocess
import shutil
from timeit import Timer
import torch
from utils.path import out_dir
from utils.pdb import (
    pdb_to_atom_backbone,
    atom_backbone_to_pdb,
    c_alpha_backbone_to_pdb,
    get_motif_mask,
    split_multi_motif_spec,
    get_motifs_and_masks_for_all_placements,
    create_evaluation_motif_pdb,
    create_evaluation_multi_motif_pdb,
)
from utils.symmetry import get_n_symmetries

logger = logging.getLogger(__name__)


def get_model(model_cfg: DictConfig) -> FrameDiffusionModel:
    if model_cfg.name not in DIFFUSION_MODEL_REGISTRY:
        raise Exception(
            f"No model called: '{model_cfg.name}'. "
            f"Choose from: {', '.join(DIFFUSION_MODEL_REGISTRY.keys())}"
        )
    model_class, model_config_resolver = DIFFUSION_MODEL_REGISTRY[model_cfg.name]
    model = model_class(**model_config_resolver(model_cfg)).to(model_cfg.device)
    return model


def check_valid_method(method: str) -> None:
    if method not in CONDITIONAL_METHOD_REGISTRY:
        raise Exception(
            f"Invalid method supplied: {method}."
            f"Valid options: {', '.join(CONDITIONAL_METHOD_REGISTRY.keys())}."
        )


@register_experiment("sample_given_motif")
def sample_given_motif(cfg: DictConfig) -> None:
    # Set-up diffusion model
    model_cfg = cfg.model
    device = torch.device(model_cfg.device)

    model = get_model(model_cfg)

    # Parse through motif problem configuration
    motif_cfg = cfg.experiment.motif
    motif_backbones = pdb_to_atom_backbone(motif_cfg.path)
    _motif_mask, mask, masked_backbones = get_motif_mask(
        motif_backbones,
        model.max_n_residues,
        motif_cfg.contig,
        return_masked_backbones=True,
    )
    motif = masked_backbones["CA"].to(device)
    motif_mask = _motif_mask.to(device)
    mask = torch.tile(mask, (cfg.experiment.n_samples, 1)).to(device)

    cond_cfg = cfg.experiment.conditional_method

    ## Currently, logic only supports the case where we have a single
    ## but possibly discontiguous motif. No multiple motifs.
    if not cfg.experiment.fixed_motif:
        if not hasattr(cond_cfg, "fixed_motif"):
            raise Exception(
                f"Conditional method '{cond_cfg.name}' does not support having a variable motif placement."
            )
        cond_cfg.fixed_motif = cfg.experiment.fixed_motif
        motif, motif_mask = get_motifs_and_masks_for_all_placements(
            mask, motif, motif_mask, motif_cfg.contig
        )

    # Choose conditional sampler
    method = cond_cfg.method
    check_valid_method(method)
    conditional_wrapper, config_resolver = CONDITIONAL_METHOD_REGISTRY[method]

    # Sample conditional on a motif being present
    setup = conditional_wrapper(model).with_config(**config_resolver(cond_cfg))
    samples = setup.sample_given_motif(mask, motif, motif_mask)

    out = out_dir()
    # Save scaffolds
    scaffolds_dir = os.path.join(out, "scaffolds_all_particles")
    os.makedirs(scaffolds_dir)
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(scaffolds_dir, f"scaffold_{i}.pdb"),
        )
    unique_scaffolds_dir = os.path.join(out, "scaffolds")
    os.makedirs(unique_scaffolds_dir)
    n_per_batch = cfg.experiment.n_samples // cond_cfg.n_batches
    for i, sample in enumerate(samples[-1][::n_per_batch]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(unique_scaffolds_dir, f"scaffold_{i}.pdb"),
        )

    # Save motif C-alpha coords, mask, and filtered PDB for evaluation
    motif_dir = os.path.join(out, "motif")
    os.makedirs(motif_dir)
    atom_backbone_to_pdb(
        {
            atom: backbone[:, _motif_mask[0] == 1]
            for atom, backbone in masked_backbones.items()
        },
        os.path.join(motif_dir, "motif_ca.pdb"),
    )
    if cfg.experiment.fixed_motif:
        torch.save(_motif_mask.cpu(), os.path.join(motif_dir, "motif_mask.pt"))
        create_evaluation_motif_pdb(
            os.path.join(motif_dir, "motif.pdb"),
            motif_cfg.path,
            _motif_mask,
            motif_cfg.contig,
        )
    else:
        # If motif is not fixed, we want to know where it is likely located.
        # At the end of each conditional method supporting this, we find
        # the motif mask that maximises whichever the log-likelihood
        # configuration was chosen. This is stored in stats/motif_mask.pt.
        path_to_likely_motif_mask = os.path.join(out, "stats", "motif_mask.pt")
        shutil.copyfile(
            path_to_likely_motif_mask,
            os.path.join(motif_dir, "motif_mask.pt"),
        )
        likely_motif_mask = torch.load(path_to_likely_motif_mask)[::n_per_batch]
        for i in range(len(likely_motif_mask)):
            create_evaluation_motif_pdb(
                os.path.join(motif_dir, f"motif_{i}.pdb"),
                motif_cfg.path,
                likely_motif_mask[i : i + 1],
                motif_cfg.contig,
            )
    with open(os.path.join(motif_dir, "motif_cfg.yaml"), "w") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in dict(motif_cfg).items()))

    # Save trace of coordinates throughout diffusion process
    if cfg.experiment.keep_coords_trace:
        traces_dir = os.path.join(out, "traces")
        os.makedirs(traces_dir)
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans[:, mask[0] == 1].detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(traces_dir, f"trace_{i}.pt"))


@register_experiment("sample_given_multiple_motifs")
def sample_given_multiple_motifs(cfg: DictConfig) -> None:
    # Set-up model
    model_cfg = cfg.model
    device = torch.device(model_cfg.device)

    model = get_model(model_cfg)

    # Parse through multi-motif problem configuration
    motif_cfg = cfg.experiment.multi_motif
    per_group_specs = split_multi_motif_spec(motif_cfg.contig)

    backbones = {}
    for motif_path in motif_cfg.paths:
        backbones[Path(motif_path).stem] = pdb_to_atom_backbone(motif_path)

    mask = None
    motifs = []
    motif_masks = []
    unmerged_motif_masks = {}
    for group_no, group_specs in per_group_specs.items():
        group_motif = torch.zeros((1, model.max_n_residues, 3))
        group_motif_mask = torch.zeros((1, model.max_n_residues))
        unmerged_motif_masks[group_no] = {}

        for motif_name, motif_specs in group_specs.items():
            motif_mask, mask, masked_backbones = get_motif_mask(
                backbones[motif_name],
                model.max_n_residues,
                motif_specs,
                return_masked_backbones=True,
            )
            group_motif += masked_backbones["CA"]
            group_motif_mask += motif_mask
            unmerged_motif_masks[group_no][motif_name] = motif_mask

        ## Make motif zero-centred. This assumes each motif group is composed
        ## of segments from only one protein. Otherwise, their centre of masses
        ## will be inconsistent with each other.
        group_motif[:, group_motif_mask[0] == 1] -= torch.mean(
            group_motif[:, group_motif_mask[0] == 1], dim=1, keepdim=True
        )
        motifs.append(group_motif)
        motif_masks.append(group_motif_mask)

    motifs = torch.cat(motifs).to(device)
    motif_masks = torch.cat(motif_masks).to(device)
    assert torch.all(
        torch.sum(motif_masks, dim=0) <= 1
    ), "Motif masks should not be intersecting."

    mask = torch.tile(mask, (cfg.experiment.n_samples, 1)).to(device)

    cond_cfg = cfg.experiment.conditional_method
    if hasattr(cond_cfg, "fixed_motif") and not cond_cfg.fixed_motif:
        raise Exception(
            f"Sampling multiple motifs currently does not support having variable motif placements."
        )

    # Choose conditional sampler
    method = cond_cfg.method
    check_valid_method(method)
    conditional_wrapper, config_resolver = CONDITIONAL_METHOD_REGISTRY[method]

    # Sample given multiple motifs
    setup = conditional_wrapper(model).with_config(**config_resolver(cond_cfg))
    samples = setup.sample_given_motif(mask, motifs, motif_masks)

    out = out_dir()
    # Save scaffolds
    scaffolds_dir = os.path.join(out, "scaffolds_all_particles")
    os.makedirs(scaffolds_dir)
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(scaffolds_dir, f"scaffold_{i}.pdb"),
        )
    unique_scaffolds_dir = os.path.join(out, "scaffolds")
    os.makedirs(unique_scaffolds_dir)
    n_per_batch = cfg.experiment.n_samples // cond_cfg.n_batches
    for i, sample in enumerate(samples[-1][::n_per_batch]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(unique_scaffolds_dir, f"scaffold_{i}.pdb"),
        )

    # Save motif C-alpha coords, mask, and (stacked) filtered PDB for evaluation
    motif_dir = os.path.join(out, "motif")
    os.makedirs(motif_dir)
    for i, backbone in enumerate(motifs):
        c_alpha_backbone_to_pdb(
            backbone[motif_masks[i] == 1], os.path.join(motif_dir, f"motif_ca_{i}.pdb")
        )
    torch.save(motif_masks.cpu(), os.path.join(motif_dir, "motif_mask.pt"))
    f_pdbs = {Path(motif_path).stem: motif_path for motif_path in motif_cfg.paths}
    create_evaluation_multi_motif_pdb(
        os.path.join(motif_dir, "motif.pdb"),
        f_pdbs,
        unmerged_motif_masks,
        per_group_specs,
    )
    with open(os.path.join(motif_dir, "motif_cfg.yaml"), "w") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in dict(motif_cfg).items()))


@register_experiment("sample_unconditional")
def sample_unconditional(cfg: DictConfig) -> None:
    # Set-up model
    model_cfg = cfg.model
    device = torch.device(model_cfg.device)

    model = get_model(model_cfg)
    model.compute_unique_only = False

    mask = torch.zeros((cfg.experiment.n_samples, model.max_n_residues), device=device)
    mask[:, : cfg.experiment.sample_length] = 1

    # Sample protein unconditionally
    ## WLOG just choose any method and call on their unconditional sample method.
    conditional_method, _ = CONDITIONAL_METHOD_REGISTRY["bpf"]
    setup = conditional_method(model)
    samples = setup.sample(mask)

    out = out_dir()
    # Save samples
    samples_dir = os.path.join(out, "samples")
    os.makedirs(samples_dir)
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans.detach().cpu(),
            os.path.join(samples_dir, f"sample_{i}.pdb"),
        )

    # Save trace of coordinates throughout diffusion process
    if cfg.experiment.keep_coords_trace:
        trace_dir = os.path.join(out, "traces")
        os.makedirs(trace_dir)
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans.detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(trace_dir, f"trace_{i}.pt"))


@register_experiment("sample_given_symmetry")
def sample_given_symmetry(cfg: DictConfig) -> None:
    # Set-up model
    model_cfg = cfg.model
    device = torch.device(model_cfg.device)

    model = get_model(model_cfg)

    n_symmetries = get_n_symmetries(cfg.experiment.symmetry)
    total_length = n_symmetries * (cfg.experiment.sample_length // n_symmetries)

    mask = torch.zeros((cfg.experiment.n_samples, model.max_n_residues), device=device)
    mask[:, :total_length] = 1

    # Choose conditional sampler
    cond_cfg = cfg.experiment.conditional_method
    method = cond_cfg.method

    ## Check method supports sampling symmetry
    conditional_wrapper, config_resolver = CONDITIONAL_METHOD_REGISTRY[method]
    if not conditional_wrapper.supports_condition_on_symmetry:
        raise Exception(
            f"Conditional method {method} does not support conditioning on symmetry."
        )
    if hasattr(cond_cfg, "fixed_motif") and not cond_cfg.fixed_motif:
        raise Exception(
            "Sampling symmetry does not support variable motif placements. "
            "Set experiment.conditional_method.fixed_motif=True."
        )

    # Sample protein conditioned on point symmetry group
    setup = conditional_wrapper(model).with_config(**config_resolver(cond_cfg))
    samples = setup.sample_given_symmetry(mask, cfg.experiment.symmetry)

    out = out_dir()
    # Save samples
    samples_dir = os.path.join(out, "samples_all_particles")
    os.makedirs(samples_dir)
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(samples_dir, f"sample_{i}.pdb"),
        )
    unique_samples_dir = os.path.join(out, "samples")
    os.makedirs(unique_samples_dir)
    n_per_batch = cfg.experiment.n_samples // cond_cfg.n_batches
    for i, sample in enumerate(samples[-1][::n_per_batch]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(unique_samples_dir, f"sample_{i}.pdb"),
        )

    # Save trace of coordinates throughout diffusion process
    if cfg.experiment.keep_coords_trace:
        trace_dir = os.path.join(out, "traces")
        os.makedirs(trace_dir)
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans.detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(trace_dir, f"trace_{i}.pt"))


@register_experiment("sample_given_motif_and_symmetry")
def sample_given_motif_and_symmetry(cfg: DictConfig) -> None:
    DASH = "-"
    # Set-up model
    model_cfg = cfg.model
    device = torch.device(model_cfg.device)

    model = get_model(model_cfg)

    # Parse through motif problem configuration
    motif_cfg = cfg.experiment.symmetric_motif
    motif_backbones = pdb_to_atom_backbone(motif_cfg.path)
    _motif_mask, mask, masked_backbones = get_motif_mask(
        motif_backbones,
        model.max_n_residues,
        motif_cfg.contig,
        return_masked_backbones=True,
    )
    motif = masked_backbones["CA"].to(device)
    motif_mask = _motif_mask.to(device)
    mask = torch.tile(mask, (cfg.experiment.n_samples, 1)).to(device)

    n_symmetries = get_n_symmetries(cfg.experiment.symmetry)
    total_oligomer_len = n_symmetries * (mask[0] == 1).sum()

    assert (
        total_oligomer_len <= model.max_n_residues
    ), f"Exceeded maximum number of residues {total_oligomer_len} > {model.max_n_residues}."
    mask[:, :total_oligomer_len] = 1

    # Choose conditional sampler
    cond_cfg = cfg.experiment.conditional_method
    method = cond_cfg.method

    ## Check method supports sampling symmetry
    conditional_wrapper, config_resolver = CONDITIONAL_METHOD_REGISTRY[method]
    if not conditional_wrapper.supports_condition_on_symmetry:
        raise Exception(
            f"Conditional method {method} does not support conditioning on symmetry."
        )
    if hasattr(cond_cfg, "fixed_motif") and not cond_cfg.fixed_motif:
        raise Exception(
            "Sampling symmetry does not support variable motif placements. "
            "Set experiment.conditional_method.fixed_motif=True."
        )

    # Sample protein conditioned on point symmetry group
    setup = conditional_wrapper(model).with_config(**config_resolver(cond_cfg))
    axes = torch.eye(3)
    for angle, axis in zip(motif_cfg.orientation, axes):
        rot = setup._general_3d_rot_matrix(
            torch.tensor([angle]) * (torch.pi / 180), axis
        )[0].to(device)
        motif = motif @ rot.T

    if cfg.experiment.fix_position:
        motif = motif + torch.tensor(motif_cfg.position, device=device).view(1, 1, 3)

    samples = setup.sample_given_motif_and_symmetry(
        mask, motif, motif_mask, cfg.experiment.symmetry, cfg.experiment.fix_position
    )

    out = out_dir()
    # Save samples
    scaffolds_dir = os.path.join(out, "scaffolds_all_particles")
    os.makedirs(scaffolds_dir)
    for i, sample in enumerate(samples[-1]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(scaffolds_dir, f"scaffold_{i}.pdb"),
        )
    unique_scaffolds_dir = os.path.join(out, "scaffolds")
    os.makedirs(unique_scaffolds_dir)
    n_per_batch = cfg.experiment.n_samples // cond_cfg.n_batches
    for i, sample in enumerate(samples[-1][::n_per_batch]):
        c_alpha_backbone_to_pdb(
            sample.trans[mask[0] == 1].detach().cpu(),
            os.path.join(unique_scaffolds_dir, f"scaffold_{i}.pdb"),
        )

    # Create multi-motif pdb for evaluation
    f_pdbs = {motif_cfg.name: motif_cfg.path}
    unmerged_motif_masks = {
        i: {
            motif_cfg.name: motif_mask.roll(
                i * (total_oligomer_len.item() // n_symmetries), 1
            )
        }
        for i in range(n_symmetries)
    }
    motif_name = motif_cfg.path.split("/")[-1].split(".")[0]
    group_temp = ",".join(
        [
            (
                f"{motif_name}/{chunk}" + "{{{group_no}}}"
                if chunk[0].isalpha()
                else chunk
            )
            for chunk in motif_cfg.contig.split(",")
        ]
    )
    per_group_specs = split_multi_motif_spec(
        ",".join(group_temp.format(group_no=i) for i in range(n_symmetries))
    )

    motif_dir = os.path.join(out, "motif")
    os.makedirs(motif_dir)
    create_evaluation_multi_motif_pdb(
        os.path.join(motif_dir, "motif.pdb"),
        f_pdbs,
        unmerged_motif_masks,
        per_group_specs,
    )
    atom_backbone_to_pdb(
        {
            atom: backbone[:, _motif_mask[0] == 1]
            for atom, backbone in masked_backbones.items()
        },
        os.path.join(motif_dir, "motif_ca.pdb"),
    )

    with open(os.path.join(motif_dir, "motif_cfg.yaml"), "w") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in dict(motif_cfg).items()))

    # Save trace of coordinates throughout diffusion process
    if cfg.experiment.keep_coords_trace:
        trace_dir = os.path.join(out, "traces")
        os.makedirs(trace_dir)
        # [K, T, N_AA, 3]
        samples_trans = torch.stack(
            [sample.trans.detach().cpu() for sample in samples]
        ).swapaxes(0, 1)

        for i, sample_trace in enumerate(samples_trans):
            torch.save(sample_trace, os.path.join(trace_dir, f"trace_{i}.pt"))


@register_experiment("evaluate_samples")
def evaluate_samples(cfg: DictConfig) -> None:
    PATH_TO_PIPELINE_SUBMODULE = "submodules/insilico_design_pipeline"

    path_to_experiment = cfg.experiment.path_to_experiment
    n_gpus = len(cfg.experiment.gpu_devices)
    n_cpus = cfg.experiment.n_cpus
    ENV_CUDA_VISIBLE_DEVICES = ",".join(map(str, cfg.experiment.gpu_devices))

    is_motif_scaffolding = os.path.isdir(os.path.join(path_to_experiment, "motif"))

    out = out_dir()

    # Create folder for generated samples/scaffolds
    pdb_dir = os.path.join(out, "pdbs")
    path_to_samples = os.path.join(
        path_to_experiment, ("scaffolds" if is_motif_scaffolding else "samples")
    )
    shutil.copytree(path_to_samples, pdb_dir)
    n_scaffolds = len(
        [name for name in os.listdir(path_to_samples) if name.endswith(".pdb")]
    )

    # Create folder for the accompanying motifs with chain indices
    # matching where it should be located in the scaffold
    if is_motif_scaffolding:
        motif_dir = os.path.join(out, "motif_pdbs")
        os.makedirs(motif_dir)

        exp_motif_dir = os.path.join(path_to_experiment, "motif")
        path_to_motif_pdb = os.path.join(exp_motif_dir, "motif.pdb")

        ## Note: file should be named the same as the scaffold
        is_fixed_motif = os.path.isfile(path_to_motif_pdb)
        if is_fixed_motif:
            for i in range(n_scaffolds):
                shutil.copyfile(
                    path_to_motif_pdb, os.path.join(motif_dir, f"scaffold_{i}.pdb")
                )
        else:
            for i in range(n_scaffolds):
                motif_placement_pdb = os.path.join(exp_motif_dir, f"motif_{i}.pdb")
                assert os.path.isfile(motif_placement_pdb)
                shutil.copyfile(
                    motif_placement_pdb, os.path.join(motif_dir, f"scaffold_{i}.pdb")
                )

        exp_motif_cfg = omegaconf.OmegaConf.load(
            os.path.join(exp_motif_dir, "motif_cfg.yaml")
        )
        motif_name = exp_motif_cfg.name

    pip_dir = os.path.join(os.getcwd(), PATH_TO_PIPELINE_SUBMODULE)
    version = "scaffold" if is_motif_scaffolding else "unconditional"
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": ENV_CUDA_VISIBLE_DEVICES})

    # Run standard/designability pipeline
    logger.info("Running designability pipeline.")
    design_pip = subprocess.Popen(
        [
            "python3",
            "pipeline/standard/evaluate.py",
            f"--version={version}",
            f"--rootdir={out}",
            f"--num_devices={n_gpus}",
            f"--num_processes={n_gpus}",
            f"--verbose",
        ],
        cwd=pip_dir,
        env=env,
    )
    design_pip.wait()

    # Run diversity pipeline
    logger.info("Running diversity pipeline.")
    diversity_pip = subprocess.Popen(
        [
            "python3",
            "pipeline/diversity/evaluate.py",
            f"--rootdir={out}",
            f"--num_cpus={n_cpus}",
        ],
        cwd=pip_dir,
        env=env,
    )
    diversity_pip.wait()

    # Profile results
    dir_to_profile = out

    ## Abide by folder format for profiling scaffolding problems by
    ## temporarily creating a directory of subdirectories to store results
    if is_motif_scaffolding:
        temp_dir = os.path.join(out, f"_temp_dir_{motif_name}")
        temp_motif_dir = os.path.join(temp_dir, f"motif={motif_name}")
        os.makedirs(temp_motif_dir)
        os.symlink(pdb_dir, os.path.join(temp_motif_dir, "pdbs"))
        os.symlink(motif_dir, os.path.join(temp_motif_dir, "motif_pdbs"))
        os.symlink(
            os.path.join(out, "info.csv"), os.path.join(temp_motif_dir, "info.csv")
        )
        dir_to_profile = temp_dir

    ## Note: if designability or diversity is zero, it throws a division by zero
    ## exception when computing for the F1 score
    logger.info("Profiling results.")
    profile_pip = subprocess.Popen(
        [
            "python3",
            f"scripts/analysis/profile_{version}.py",
            f"--rootdir={dir_to_profile}",
        ],
        cwd=pip_dir,
        env=env,
    )
    profile_pip.wait()

    ## Remove temporary directory
    if is_motif_scaffolding:
        shutil.rmtree(temp_dir)


@register_experiment("debug_gpu_stats")
def debug_gpu_stats(cfg: DictConfig) -> None:
    exp_cfg = cfg.experiment
    model_cfg = cfg.model

    device = torch.device(model_cfg.device)

    model = get_model(model_cfg)
    model.n_timesteps = exp_cfg.n_trials

    mask = torch.zeros((exp_cfg.n_samples, model.max_n_residues), device=device)
    mask[:, : model.max_n_residues] = 1

    conditional_method, _ = CONDITIONAL_METHOD_REGISTRY["bpf"]
    setup = conditional_method(model)
    model.compute_unique_only = False

    # Check optimal batch size for hardware
    for batch_size in exp_cfg.batch_size_range:
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(device)

        model.batch_size = batch_size
        timer = Timer(lambda: setup.sample(mask))

        sample_time = timer.timeit(number=1) / (exp_cfg.n_trials * exp_cfg.n_samples)
        max_memory = torch.cuda.max_memory_allocated(device) / float(1 << 30)

        logger.info(
            f"batch_size: {batch_size}, sample_time: {sample_time:.8f}s, max_memory: {max_memory:.4f}GB"
        )
