# Keras Deep Learning Pipeline

This repository is intended to streamline the training of deep learning models.
- Training data is inventoried and partitioned into training/validation sets for reproducibility.
- All training hyperparameters are collected in a JSON file which is exposed to the user.
- Trained models can be applied to unlabelled data for prediction
- New models and functionality can be integrated into this pipeline by following defined specifications.

## Quickstart (Ubuntu/Python)

1. This pipeline was developed for Python 3.10 on Ubuntu 22.04 (WSL). Operating behavior may vary in other environments.
2. Clone repository files to local machine and navigate to project directory.
3. Ensure that the required system packages are installed and updated:
    - `sudo apt-get install python3-pip python3-venv`
    - `pip3 install --upgrade pip`
4. Create virtual environment in which to install libraries.
    - `python3 -m venv .venv`
    - `source .venv/bin/activate`
5. Install the required Python libraries into the virtual environment.
    - `pip3 install -r requirements.txt`
6. Run main script
    - `python3 main.py --config config.json`


## Config Parameters

TODO

## Pipeline Development Guide

TODO

## File Overview

TODO