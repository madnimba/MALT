#!/bin/bash

python connectomics_ctrl.py stop
nvidia-smi   # should show only GUI processes
python generate_trajectories.py
python connectomics_ctrl.py resume