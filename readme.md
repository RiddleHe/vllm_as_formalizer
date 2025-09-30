# Vision Language Models Cannot Plan, but Can They Formalize?

## Overview

This repo implements the benchmarks and pipeline logics for this paper.

## Benchmark structure

Two benchmarks are available (adding support for ALFRED currently). 
They have the following structure:

```
data/
  dataset_root/
    subtask_name/
      observations/  $ contain .png files
      domain.pddl
      instruction.txt
      problem.pddl
      plan.txt
```

## Usage

We use `scripts/main.py` as the entry point to choose a dataset and a pipeline to evaluate.

Command line arguments include:
- `--dataset_dir`
- `--dataset`
- `--model`
- `--pipeline`


Example:
```
python main.py --dataset_dir ... --dataset ... --model ... --pipeline ...
```

## Results
- Output directory structure:
```
results/
  <dataset>_<pipeline>_<model>/
    subtask_name/
      subtask_name.pddl
```