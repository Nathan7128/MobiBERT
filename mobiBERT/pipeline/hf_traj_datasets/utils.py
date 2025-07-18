"""
This Python module contains variables, functions, and other elements that may be useful for modules included in the "hf_datasets" part of the library.
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import os

# Internal libraries

import sys
sys.path.append("mobiBERT")

from pipeline.utils import library_data_dir

# --------------------------------------------------------- VARIABLE DEFINITIONS -----------------------------------------------------------------------------------------------------------------------

datasets_dir = os.path.join(library_data_dir, "hf_traj_dataset")
"""Path to the HuggingFaceTrajDataset's storage directory."""

hf_dataset_dirname = "hf_dataset"
"""Name of the directory containing the Hugging Face dataset."""

extra_info_filename = "extra_info.json"
"""Name of the file containing extra information about the HuggingFaceTrajDataset instance."""