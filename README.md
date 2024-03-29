# Sagittarius
Gene expression time-series extrapolation for heterogeneous data

## Introduction
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
|       | --- dataset/: files for the EvoDevo dataset.
|
| --- LINCS/: experimental code for the LINCS dataset, including:
|       | --- simulate_new_LINCS_measurements.ipynb: notebook to simulate a new LINCS dataset at unmeasured dose
|                                                    and time for all combinations of compounds and cell lines.
|
| --- TCGA/: experimental code for the TCGA dataset, including:
|       | --- simulate_new_TCGA_measurements.ipynb: notebook to simulate new TCGA mutation dataset at unobserved
|                                                   survival times for all cancer types.
```


## Installation Tutorial

### System Requirements
Sagittarius is implemented using Python 3.9 on LINUX. Sagittairus expects torch==1.9.1+cu11.1, numpy==1.21.2, pandas==1.3.3, scikit-learn=0.24.2, matplotlib==3.4.3, seaborn==0.11.2, umap-learn=0.5.1, anndata=0.8.0, statsmodels==0.13.0, tqdm==4.62.3, and so on. For best performance, Sagittarius can be run on a GPU. However, all experiments can also be run on a CPU by not setting the `--gpu` flag. Typical installation requires approximately 5 minutes.

## How to use our code
To begin, update the settings in `config.py` to reflect your environment. The EvoDevo dataset files are also included in the repository, so EvoDevo experiments can be run as a self-contained example of the model (see 2, 3).

1. To reproduce figures from the paper, open the respective `figures/reproduce_figure_X.ipynb` or `figures/reproduce_supplementary_figure_Y.ipynb` notebook and run.
2. To run the EvoDevo extrapolation task, run `python EvoDevo/Sagittarius_EvoDevo_extrapolation_experiment.py`. This will output (to std out by default) the sequence-level extrapolation performance summary for all time series in the EvoDevo dataset. To view all possible commandline arguments, add the `--help` flag. To set gpu, use `--gpu <gpu number>`. To load the paper's pretrained model, use `--reload --model-file EvoDevo/trained_models/Sagittarius_extrapolation_model.pth --preload-gene-mask EvoDevo/trained_models/Sagittarius_paper_genemask.txt`. To mimic the paper's results, use `--config EvoDevo/model_config_files/Sagittarius_config.json`. Running inference-only (with the --reload flag) on a standard laptop requires approximately 1 minute; training & inference requires approximately 1 hour. 
3. To simulate a new dataset from the fully-trained EvoDevo model, open the `EvoDevo/simulate_new_EvoDevo_measurements.ipynb` and run, replacing the time points to simulate as desired. A dataset made up of the default time points can also be found at: https://figshare.com/projects/Sagittarius/144771.
4. To run the LINCS extrapolation tasks, run `python LINCS/Sagittarius_LINCS_<taskname>_experiment.py`. To view all possible commandline arguments, add the `--help` flag. To set gpu, use `--gpu <gpu number>`. To load the paper's pretrained model, use `--reload --model-file LINCS/trained_models/Sagittarius_<taskname>_model.pth`. To mimic the paper's results, use `--config LINCS/model_config_files/Sagittarius_config.json`.
5. To simulate a new dataset from the fully-trained LINCS model, open the `LINCS/simulate_new_LINCS_measurements.ipynb` and run, replacing the dose and time points to simulate as desired.
6. To run the TCGA patient-by-patient extrapolation task, run `python TCGA/Sagittarius_patient_by_patient_extrapolation.py`. To view all possible commandline arguments, add the `--help` flag. To set gpu, use `--gpu <gpu number>`. Set the cancer type of interest with `--cancer-type <SARC, THCA, etc>`. Determine the number of intersecting cancer types non-stationary genes with `--non-stationary-k <k>`. To reproduce the results from the paper, use `--non-stationary-k 4` with `--cancer-type SARC`, and `--non-stationary-k 2` with `--cancer-type THCA`.
7. To simulate a new dataset with early-stage cancer patients, open the `TCGA/simulate_new_TCGA_measurements.ipynb` and run, replacing the survival time points to simulate as desired. A dataset made up of the default survival time points to simulate can also be found at: https://figshare.com/projects/Sagittarius/144771.
