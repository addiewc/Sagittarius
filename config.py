"""
Configuration file for Sagittarius.
"""

SAGITTARIUS_LOC = ''  # put the location of the Sagittarius directory here

# Evo-Devo related file locations
EVO_DEVO_DATA_LOC = SAGITTARIUS_LOC + 'EvoDevo/dataset/'  # put EvoDevo dataset files here
EVO_DEVO_GENE_MAPPING_FILE = SAGITTARIUS_LOC + 'EvoDevo/dataset/name_mapping.txt'

# LINCS related file locations
LINCS2D_DATA_LOC = ''  # put LINCS dataset files here

# TCGA related file locations
TCGA_DATA_LOC = ''  # put TCGA dataset files here
NON_STATIONARY_GENES_DIR = SAGITTARIUS_LOC + 'TCGA/non_stationary_gene_masks/'  # put non-stationary masks here
FILTERING_MASK_LOCATION = SAGITTARIUS_LOC + 'TCGA/filtering_masks/'  # put filtered dataset masks here
