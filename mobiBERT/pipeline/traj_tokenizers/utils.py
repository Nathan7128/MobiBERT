"""
This Python module contains variables, functions, and other elements that may be useful for modules included in the "traj_tokenizer" part of the library.
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import os
import sys
sys.path.append("mobiBERT")

from datasets import Dataset

# Internal libraries

from pipeline.utils import library_data_dir

# --------------------------------------------------------- VARIABLE DEFINITIONS -----------------------------------------------------------------------------------------------------------------------

tokenizers_dir = os.path.join(library_data_dir, "traj_tokenizers")
"""Path to the TrajTokenizer's storage directory."""

hf_tokenizer_dirname = "hf_tokenizer"
"""Name of the directory containing the Hugging Face tokenizer."""

extra_info_filename = "extra_info.json"
"""Name of the file containing extra information about the TrajTokenizer instance."""

# --------------------------------------------------------- FUNCTION IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

def batch_iterator(batch_size=1000, dataset: Dataset = None) :
    """Returns the "text" data of the dataset by batch, allowing for memory relief.

    Args:
        batch_size (int, optional): Number of sequence returned per call of this function. Defaults to 1000.
        dataset (Dataset, optional): Hugging Face dataset containing the data to process. Defaults to None.
    """
    for i in range(0, len(dataset), batch_size):
        # This yields a list of text examples.
        yield dataset[i : i + batch_size]["text"]