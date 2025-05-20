# ARC Challenge Project

This repository contains code for working with the Abstraction and Reasoning Corpus (ARC) challenge.

## Components

- **ARC Data Loader**: A simple loader for ARC task data (`arc_data_loader.py`)
- **Task Visualization**: Tools for visualizing ARC tasks using matplotlib (`visualize_tasks.py`)
- **Model Integration**: Utilities for loading and using language models with ARC (`load_model.py`)

## Getting Started

The project uses the ARC dataset structure with training and evaluation examples.

```python
# Example: Load training data
from arc_data_loader import ARCDataLoader

training_data = ARCDataLoader("/path/to/training/data")
```

## Requirements

- Python 3.x
- numpy
- matplotlib
- transformers (for model integration)

## Dataset

The ARC dataset should be organized in the following structure:
```
data/
  training/
    *.json
  evaluation/
    *.json
```

Each JSON file contains training and test examples for a specific task.
