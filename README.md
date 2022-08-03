# Sagittarius
Gene expression time-series extrapolation for heterogeneous data

## Introductoin
Sagittarius is a model for temporal gene expression extrapolation simulate unmeasured gene expression data from unaligned, heterogeneous time series data. This is a python repository to simulate transcriptomic profiles at time points outside of the range of time points available in the measured data.

## Repository structure

```
| Sagittarius/
|
| --- figures/: jupyter notebooks to recreate the figures in the paper.
|
| --- models/: Sagittarius model file and wrapper to facilitate easy interaction.
|
| --- EvoDevo/: experimental code for the Evo-Devo dataset, including:
|       | --- simulate_new_EvoDevo_measurements.ipynb: notebook to simulate a new EvoDevo dataset at unmeasured
|                                                      time points for all species and organ combinations.
|
| --- LINCS/: experimental code for the LINCS dataset, including:
|       | --- simulate_new_LINCS_measurements.ipynb: notebook to simulate a new LINCS dataset at unmeasured dose
|                                                    and time for all combinations of compounds and cell lines.
```


## Installation Tutorial

### System Requirements
Sagittarius is implemented using Python 3.9 on LINUX. Sagittairus expects torch==1.9.1+cu11.1, numpy, pandas, sklearn, matplotlib, seaborn, and so on. For best performance, Sagittarius can be run on a GPU. However, all experiments can also be run on a CPU by not setting the `--gpu` flag.

## How to use our code
1. To reproduce figures from the paper, open the respective `figures/reproduce_figure_X.ipynb` or `figures/reproduce_supplementary_figure_Y.ipynb` notebook and run.
2. To run the EvoDevo extrapolation task, run `python EvoDevo/Sagittarius_EvoDevo_extrapolation_experiment.py`. To view all possible commandline arguments, add the `--help` flag. To set gpu, use `--gpu <gpu number>`. To load the paper's pretrained model, use `--reload --model-file EvoDevo/trained_models/Sagittarius_extrapolation_model.pth`. To mimic the paper's results, use `--config EvoDevo/model_config_files/Sagittarius_config.json`.
3. To simulate a new dataset from the fully-trained EvoDevo model, open the `EvoDevo/simulate_new_EvoDevo_measurements.ipynb` and run, replacing the time points to simulate as desired. A dataset made up of the default time points can also be found at: https://figshare.com/projects/Sagittarius/144771.
4. To simulate a new dataset from the fully-trained LINCS model, open the `LINCS/simulate_new_LINCS_measurements.ipynb` and run, replacing the dose and time points to simulate as desired.
