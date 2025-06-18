# pibi-sfr

This repository contains the code for the paper:
    
S. Damiano and T. van Waterschoot, "Sound Field Reconstruction Using Physics-Informed Boundary Integral Networks", in *Proceedings of the 33rd European Signal Processing Conference (EUSIPCO 2025)*, accepted paper, Isola delle Femmine, Italy, Sep. 2025. 

You can find the paper here: [10.48550/arXiv.2506.03917](https://doi.org/10.48550/arXiv.2506.03917).

## Usage

### Installation
The code runs in a Conda virtual environment, that can be created from the provided yml file:
    
    conda env create -f environment.yml
    
### Running the code
The experiments using the PINN and PIBI models can be run from the scripts ```pibi_experiments.py``` and ```pinn_experiments.py```. These scripts can be called from the command line, providing a seed for the random number generator (default: 42).

The PIDL experiments rely on code available at [1]

### External references
[1] The code that implements the PIDL method is available at: [https://github.com/steDamiano/PIDL-sound-field-reconstruction](https://github.com/steDamiano/PIDL-sound-field-reconstruction)

[2] The original PIBI-Net code, from which this work was inspired, is available at: [https://github.com/MonikaNagy-Huber/PIBI-Net](https://github.com/MonikaNagy-Huber/PIBI-Net)

## Acknowledgements
This project has received funding from KU Leuven internal funds C3/23/056 and from FWO Research Project G0A0424N. The resources and services used in this work were provided by the VSC (Flemish Supercomputer Center), funded by the Research Foundation - Flanders (FWO) and the Flemish Government.