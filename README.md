# MLTools

A collection of Python-based tools and utilities built on PyTorch to streamline machine learning workflows.

## Installation

Clone this repository and install dependencies from requirements.txt.
Alternatively, install via pyproject.toml using pip, since it references the same requirements file.

## Usage

There are no script in this project. Instead, import the modules and classes you need.

mltools/
├── docker                   # Docker-related files
│   └── Dockerfile           # Docker configuration file
├── LICENSE                  # License file
├── mltools                  # Main package directory
│   ├── attention.py         # Attention mechanisms
│   ├── bayesian.py          # Bayesian methods
│   ├── clustering.py        # Clustering algorithms
│   ├── cnns.py              # Convolutional Neural Networks
│   ├── deepset.py           # Deep set operations
│   ├── diffusion.py         # Diffusion models
│   ├── flows.py             # Flow-based models
│   ├── hydra_utils.py       # Hydra configuration utilities
│   ├── lightning_utils.py   # PyTorch Lightning utilities
│   ├── loss.py              # Loss functions
│   ├── mlp.py               # Multi-Layer Perceptrons
│   ├── modules.py           # Custom neural network modules
│   ├── numpy_utils.py       # NumPy utilities
│   ├── optimisers.py        # Optimizers
│   ├── plotting.py          # Plotting utilities
│   ├── schedulers.py        # Learning rate schedulers
│   ├── torch_utils.py       # PyTorch utilities
│   ├── transformers.py      # Transformer models
│   └── utils.py             # General utilities
├── pyproject.toml           # Project configuration file
├── README.md                # Readme file
├── requirements.txt         # Dependencies file
└── tests                    # Unit tests directory
    ├── test_cnns.py         # Tests for CNNs
    ├── test_flows.py        # Tests for flow-based models
    ├── test_loss.py         # Tests for loss functions
    ├── test_mlp.py          # Tests for MLPs
    ├── test_modules.py      # Tests for custom modules
    ├── test_optimisers.py   # Tests for optimizers
    └── test_transformers.py # Tests for transformers

## Docker

A ready-to-use container setup can be found in `Dockerfile`.

## Tests

Unit tests are in `tests`. Run them via PyTest after installation:

```bash
pytest
```

## License

This project is licensed under the MIT License.
