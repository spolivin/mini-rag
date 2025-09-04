#!/bin/bash

# TO BE RUN WITH `source` command: "source setup_env.sh"

# Create a new conda environment
conda create -y -n rag-venv python=3.10
conda activate rag-venv

pip install -r requirements.txt
