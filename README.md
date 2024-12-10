## Diffusion Posterior Sampling via SMC for Zero-Shot Scaffolding of Protein Motifs

This repository provides an interface for solving motif scaffolding problems with an unconditional diffusion model as a prior. It defines inverse problems for a variety of tasks and solves them by sampling the posterior via sequential Monte Carlo. This permits conditional sampling without additional training to the chosen unconditional model.

The following are the supported tasks, samplers, and likelihood formalisations for conditioning on a motif.

### Motif Scaffolding Tasks
- (Single-) motif scaffolding
- Multi-motif scaffolding
- Symmetric motif scaffolding
- Inpainting/scaffolding with degrees of freedom

### (SMC-Aided) Diffusion Posterior Sampling Methods
- Bootstrap Particle Filter
- SMCDiff [(Trippe _et al._, 2023)](https://arxiv.org/pdf/2206.04119)
- Filtering Posterior Sampling [(Dou & Song, 2024)](https://openreview.net/pdf?id=tplXNcHZs1)
- Twisted Diffusion Sampler [(Wu _et al._, 2023)](https://arxiv.org/pdf/2306.17775)
- Monte Carlo Guided Diffusion [(Cardoso _et al._, 2023)](https://openreview.net/pdf?id=nHESwXvxWK)

### Motif Scaffolding Guidance Potentials/Likelihoods
- Masking
    - Condition on a partial view of the backbone with a fixed orientation.
- Distance
    - Condition on pairwise residue distances.
- Frame distance
    - Condition on pairwise residue distances and rotation matrix deviations from the frame representation.
- Frame-aligned point error
    - Condition on several partial views of the backbone, each aligned to meet the motif at a different residue index.
- (Root mean) squared deviation
    - Condition on a partial view of the backbone, oriented to have minimal deviation from the motif.

Additionally, other unconditional models can be supported by creating an adapter for them in `model`, registering them and their parameters, and adding a config in `config/model`. Currently, we have [Genie-SCOPe-128 and Genie-SCOPe-256](https://github.com/aqlaboratory/genie) available. The conditional samplers assume the frame representation of the protein, so some extra engineering may be required for other models.

## Installation

To match our setup, use Python 3.9 with CUDA version 11.8 and above. First, pip install the requirements.
```bash
pip install -r requirements.txt
```
Then, initialise the submodules if not already setup.
```bash
git submodule update --init
```
Finally, but optionally, we use the [insilico design pipeline](https://github.com/aqlaboratory/insilico_design_pipeline/tree/main) from AQLaboratory for evaluation. Run their bash scripts for installing TMScore, ProteinMPNN, and ESMFold to set up the self-consistency pipeline.

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
│   ├── resampling.py                   # for low-variance resampling
│   └── symmetry.py                     # for dealing with symmetry
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
|`evaluate_samples`| Evaluate motif scaffolding results using insilico design pipeline. |

### Examples

Sample 16 proteins with 96 residues each using unconditional model Genie-SCOPe-128 (default if unspecified) on GPU device #1.
```bash
python3 main.py experiment=sample_unconditional \
    experiment.n_samples=16 \
    experiment.sample_length=96 \
    model=genie-scope-128 \
    model.device=cuda:1
```

Scaffold motif problem 3IXT using TDS with masking likelihood, twist scale=2.0, and K=8 particles.
```bash
python3 main.py experiment=sample_given_motif \
    experiment/motif=3IXT \
    experiment/conditional_method=tds-mask \
    experiment.conditional_method.twist_scale=2.0 \
    experiment.n_samples=8
```

Produce 16 scaffolds for motif problem 1PRW using TDS with distance likelihood and K=8 particles.
```bash
python3 main.py experiment=sample_given_motif \
    experiment/motif=1PRW \
    experiment/conditional_method=tds-distance \
    experiment.conditional_method.n_batches=16
    experiment.n_samples=128
```

Scaffold motif problem 5TPN, allowing the motif to be placed anywhere, using TDS with masking likelihood and K=8 particles.
```bash
python3 main.py experiment=sample_given_motif \
    experiment/motif=5TPN \
    experiment.fixed_motif=False \
    experiment/conditional_method=tds-mask \
    experiment.n_samples=8
```

Scaffold multi-motif problem 1PRW_two using TDS with frame-based distance likelihood and K=8 particles.
```bash
python3 main.py experiment=sample_given_multiple_motifs \
    experiment/multi_motif=1PRW_two \
    experiment/conditional_method=tds-frame-distance \
    experiment.n_samples=8 \
    model=genie-scope-256
```

Sample a monomer with 250 residues and C-5 internal symmetry using FPS-SMC with K=16 particles.
```bash
python3 main.py experiment=sample_given_symmetry \
    model=genie-scope-256 \
    experiment.sample_length=250 \
    experiment.symmetry=C-5 \
    experiment/conditional_method=fpssmc \
    experiment.n_samples=16
```

Evaluate samples from unconditional, single-motif scaffolding, or multi-motif scaffolding experiments using the insilico design pipeline with CUDA visible devices #0, #2, and #3
```bash
python3 main.py experiment=evaluate_samples \
    experiment.path_to_experiment=<path_to_hydra_output_folder> \
    experiment.gpu_devices=\[0, 2, 3\]
```

Each of the conditional methods and models have their own default hyperparameters which can be overwritten in the command-line. Check out their config files for more info. Custom motifs can also be scaffolded by creating a config file in `configs/experiment/motif` following the specification of configs in that directory. The case is similar with multiple motifs, except they are stored in `configs/experiment/multi_motif`.
