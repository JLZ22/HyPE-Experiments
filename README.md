# HyPE_Experiments

This repository contains implementations of baselines and testbeds for different meta-RL algorithms. The primary purpose is to test the performance of Hypothesis Network Planned Exploration with different meta-RL algorithms and compare it with the baselines. 

## Installation

Install dependencies with the following command:

```bash
pip install -r requirements.txt
```

By default, PyTorch is installed without CUDA support. If you want to use CUDA, first uninstall with 

```bash
pip uninstall torch torchvision torchaudio
```

Then, re-install PyTorch with CUDA support by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

## Usage

### Experiments
To run any of the experiments in the `./experiments` directory, navigate to the root directory of the project and run the following command:

```bash
python3 -m experiments.<experiment name>
```

<details>
<summary>Example</summary>
If you want to run `foo.py`, run the following command:

```bash
python3 -m scripts.foo
```
</details>

### Tests
To run the tests, navigate to the root directory of the project and run the following command:

```bash
pytest
```

For logging, you can use the following command:

```bash
pytest --log-cli-level=INFO
```

The hierarchy of logging levels from least to most verbose is `CRITICAL`, `ERROR`, `WARNING`, `INFO`, `DEBUG`.