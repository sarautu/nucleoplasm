# Nucleoplasm simulations

This repository contains the Dedalus simulation scripts associated with the paper:

S. Alex Rautu, Alexandra Zidovska, David Saintillan, and Michael J. Shelley,  
**Active Hydrodynamic Theory of Euchromatin and Heterochromatin**,  
*Physical Review X* **16**, 021009 (2026).  
DOI: `10.1103/8n8h-7gx6`

The code implements continuum hydrodynamic simulations of euchromatin, heterochromatin, and the surrounding nucleoplasmic fluid. The scripts solve the governing equations in two geometries:

- `box.py`: periodic Cartesian box geometry.
- `sphere.py`: spherical nuclear geometry.

The simulations require Dedalus v3 and are intended to be run in a Python/MPI environment.

## Repository contents

```text
nucleoplasm/
├── box.py
├── sphere.py
├── README.md
├── LICENSE
└── .gitignore
```

Only the source scripts and repository documentation are included. Large simulation outputs are not stored in the repository.

## Requirements

These scripts require a working Dedalus v3 installation. Dedalus should be installed by following the official installation instructions:

<https://dedalus-project.readthedocs.io/en/latest/pages/installation.html>

The scripts should be run from an activated Dedalus environment, for example:

```bash
conda activate <your-dedalus-environment>
```

When Dedalus is installed following the official instructions, the required Python dependencies and compiled libraries, including MPI, FFTW, and HDF5, are handled by that environment.

## Running the simulations

Clone the repository:

```bash
git clone https://github.com/sarautu/nucleoplasm.git
cd nucleoplasm
```

### Box geometry

To run the Cartesian box simulation with the default parameters:

```bash
mpiexec -n 128 python3 box.py
```

To specify the mean euchromatin density, mean heterochromatin density, and active forcing parameter:

```bash
mpiexec -n 128 python3 box.py 24 24 0
```

The three command-line arguments are:

```text
python3 box.py <nE_bar> <nH_bar> <Alpha>
```

For example, `24 24 0` sets:

```text
nE_bar = 24
nH_bar = 24
Alpha  = 0
```

To restart from the latest checkpoint:

```bash
mpiexec -n 128 python3 box.py 24 24 0 --restart
```

To restart from a specific checkpoint number:

```bash
mpiexec -n 128 python3 box.py 24 24 0 --restart 5
```

### Spherical geometry

To run the spherical simulation with the default parameters:

```bash
mpiexec -n 128 python3 sphere.py
```

To specify the mean euchromatin density, mean heterochromatin density, and active forcing parameter:

```bash
mpiexec -n 128 python3 sphere.py 24 48 0
```

The three command-line arguments are:

```text
python3 sphere.py <nE_bar> <nH_bar> <Alpha>
```

For example, `24 48 0` sets:

```text
nE_bar = 24
nH_bar = 48
Alpha  = 0
```

To restart from the latest checkpoint:

```bash
mpiexec -n 128 python3 sphere.py 24 48 0 --restart
```

To restart from a specific checkpoint number:

```bash
mpiexec -n 128 python3 sphere.py 24 48 0 --restart 5
```

The number of MPI processes can be adjusted depending on available computational resources.

## Output files

The scripts write Dedalus output files to directories named according to the simulation parameters. Typical output directories include:

```text
fields_nE_<nE>_nH_<nH>_Alpha_<Alpha>/
flows_nE_<nE>_nH_<nH>_Alpha_<Alpha>/
order_nE_<nE>_nH_<nH>_Alpha_<Alpha>/
checkpoints_nE_<nE>_nH_<nH>_Alpha_<Alpha>/
```

For default runs without command-line parameters, the corresponding directories are:

```text
fields/
flows/
order/
checkpoints/
```

These generated files are not included in the repository and should not normally be committed to GitHub.

## Notes on reproducibility

The numerical parameters, spatial resolutions, timestepping choices, initial conditions, and output cadences are specified directly in `box.py` and `sphere.py`.

The scripts are intended to document and reproduce the simulations reported in the associated paper. Users interested in modifying the simulations should begin by inspecting the parameter blocks near the top of each script.

## Citation

If you use this code, please cite:

S. Alex Rautu, Alexandra Zidovska, David Saintillan, and Michael J. Shelley,  
**Active Hydrodynamic Theory of Euchromatin and Heterochromatin**,  
*Physical Review X* **16**, 021009 (2026).  
DOI: `10.1103/8n8h-7gx6`

## License

This code is released under the Apache License, Version 2.0. See `LICENSE`.
