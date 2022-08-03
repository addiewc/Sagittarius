# Sagittarius
Gene expression time-series extrapolation for heterogeneous data

### Introductoin
Sagittarius is a model for temporal gene expression extrapolation simulate unmeasured gene expression data from unaligned, heterogeneous time series data. This is a python repository to simulate transcriptomic profiles at time points outside of the range of time points available in the measured data.

### Repository structure

figures/ -- jupyter notebooks to recreate the figures in the paper.
models/ -- Sagittarius model file and wrapper to facilitate easy interaction.
EvoDevo/ -- experimental code for the Evo-Devo dataset, including `simulate_new_EvoDevo_measurements.ipynb` notebook to simulate a new EvoDevo dataset at unmeasured time points for all species and organ combinations.
