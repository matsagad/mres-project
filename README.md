# MRes Project

Ongoing project for conditioning on protein diffusion models.

## Usage

The project uses the [Hydra](https://github.com/facebookresearch/hydra) framework for handling different experimental setups. Configuration files and groups are defined under the `config` folder.

To get started, the following command will show the available options for config groups, e.g. an experiment type, as well as the default parameters set.
```{bash}
python3 main.py --help
```