# MRes Project

This repository provides an interface for solving motif scaffolding problems. It defines inverse problems for a variety of tasks and performs diffusion posterior sampling via sequential Monte Carlo to solve them. This permits conditional sampling without additional training to an unconditional diffusion model for protein backbones.

The following are the supported tasks, samplers, and likelihood formalisations for conditioning on a motif.

### Motif Scaffolding Tasks
- (Single-) motif scaffolding
- Multi-motif scaffolding
- Symmetric motif scaffolding
- Inpainting/scaffolding with degrees of freedom

### (Diffusion) Posterior Sampling Methods
- Bootstrap Particle Filter
- SMCDiff [(Trippe _et al._, 2023)](https://arxiv.org/pdf/2206.04119)
- Filtering Posterior Sampling [(Dou & Song, 2024)](https://openreview.net/pdf?id=tplXNcHZs1)
- Twisted Diffusion Sampler [(Wu _et al._, 2023)](https://arxiv.org/pdf/2306.17775)

### Motif Scaffolding Likelihood Formalisations
- Masking (as done by previous methods)
    - Condition on a partial view of the backbone with a fixed orientation.
- Distance
    - Condition on the pairwise residue distances.
- Frame-based distance
    - Condition on the pairwise residue distances and pair-wise residue rotation matrix deviations from the rigid body frame representation.

## Structure

```
.
├── conditional/                    # Posterior samplers
│   ├── __init__.py                     # Sampler registration
│   ├── components/                     # Reusable components
│   │   ├── observation_generator.py        # for generating y sequence
│   │   └── particle_filter.py              # for filtering
│   ├── bpf.py                          # Bootstrap particle filter
│   ├── fpssmc.py                       # Filtering posterior sampling
│   ├── smcdiff.py                      # SMCDiff
│   ├── tds.py                          # Twisted diffusion sampler
│   └── wrapper.py                      # Abstract class for samplers
├── config/                         # Configs
│   ├── config.yaml                     # Main config file
│   ├── experiments/                    # Experiment config groups
│   │   └── ...
│   └── model/                          # Model config groups
│       └── ...
├── data/                           # Motif scaffolding data   
│   ├── motif_problems/                 # RFDiffusion benchmark
│   │   └── ...
│   ├── multi_motif_problems/           # Genie2 benchmark
│   │   └── ...
│   └── symmetric_motif_problems/       # RFDiffusion trimeric covid binder
│       └── ...
├── experiments/                    # Experiments
│   ├── __init__.py                     # Experiment registration
│   └── experiments.py                  # Experiment definitions
├── model/                          # Diffusion model
│   ├── __init__.py                     # Model registration
│   ├── diffusion.py                    # Abstract class for diffusion models
│   └── genie.py                        # Genie adapter
├── multirun/                       # Output of multirun/sweeping experiments
│   └── ...
├── outputs/                        # Output of experiments
│   └── ...
├── protein/                        # Protein-related functions
│   └── frames.py                       # Abstract class for frames
├── scripts/                        # Scripts for config generation, etc.
│   └── ...
├── submodules/                     # Git submodules
│   └── ...
├── utils/                          # Utility functions
│   ├── path.py                         # for resolving paths
│   ├── pdb.py                          # for working with PDBs
│   ├── registry.py                     # for handling registrations
│   └── resampling.py                   # for low-variance resampling
└── main.py                         # Main entry point
```

## Usage

The project uses the [Hydra](https://github.com/facebookresearch/hydra) framework for handling different experimental setups. Configuration files and groups are defined under the `config` folder.

To get started, the following command will show the available options for config groups, e.g. an experiment type, as well as the default parameters set.
```{bash}
python3 main.py --help
```

### Supported Experiments
Configured through the option `experiment={experiment_name}`. Check their arguments and defaults in `config/experiments`. 

| Experiment | Description |
|:---:|---|
|`sample_unconditional`| Sample unconditional samples from the diffusion model. Total length must be specified. |
|`sample_given_motif`| Sample conditioned on a motif being present in the samples. Motif config files have specifications like in RFDiffusion. |
|`sample_given_multiple_motifs`| Sample conditioned on multiple motifs being present on the samples. Motif config files have specifications like in Genie2. |
|`sample_given_symmetry`| Sample conditioned on the samples following a point symmetry. |
|`sample_given_motif_and_symmetry`| Sample conditioned on a motif being present in the samples and them following a point symmetry. Motif specification is for a single monomer. |

