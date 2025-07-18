"""
This Python module contains variables, functions, and other elements that may be useful for modules included in the "models" part of the library.
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import os

from transformers import PretrainedConfig
from transformers.trainer_utils import EvalPrediction

# Internal libraries

import sys
sys.path.append("mobiBERT")

from pipeline.utils import library_data_dir
from pipeline.traj_tokenizers.tokenizers import TrajTokenizer
from pipeline.traj_models.traj_metrics import TrajMetric

# --------------------------------------------------------- VARIABLE DEFINITIONS -----------------------------------------------------------------------------------------------------------------------

models_dir = os.path.join(library_data_dir, "traj_models")
"""Directory containing all data related to the TrajModels."""

pre_trained_models_dir = os.path.join(models_dir, "pre_trained_models")
"""Directory containing all data related to the TrajPreTrained models."""

fine_tuned_models_dir = os.path.join(models_dir, "fine_tuned_models")
"""Directory containing all data related to the TrajFineTuned models."""

hf_model_dirname = "hf_model"
"""Name of the directory containing the Hugging Face model associated with a TrajModel."""

extra_info_filename = "extra_info.json"
"""Name of the file containing extra information about the TrajModel."""

# --------------------------------------------------------- FUNCTION IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

def update_special_tokens_ids(tokenizer: TrajTokenizer, config: PretrainedConfig):
    """Assigns the IDs of the special tokens from the associated Tokenizer to a PretrainedConfig.

    This method returns the updated PretrainedConfig.

    Args:
        tokenizer (TrajTokenizer): TrajTokenizer associated with the model.
        config (PretrainedConfig): Hugging Face configuration of the model.
    """

    # Vocabulary (dictionary) of the tokenizer ("token": "id")
    tokenizer_vocab = tokenizer.get_hf_tokenizer().get_vocab()
    # Dictionary mapping special token class attributes to their values
    special_tokens_map = tokenizer.get_hf_tokenizer().special_tokens_map

    # List of all special tokens handled by the model
    config_special_tokens = [attr for attr in config.to_dict().keys() if "token_id" in attr]

    # Update the special token IDs in the config if they are also handled by the tokenizer
    for special_token in config_special_tokens:
        # Name assigned by the tokenizer to the current special token
        tokenizer_special_token = special_token.replace("_id", "")
        if tokenizer_special_token in special_tokens_map.keys():
            config.update({special_token: tokenizer_vocab[special_tokens_map[tokenizer_special_token]]})

    return config



def get_compute_metrics_fn(eval_metrics: list[TrajMetric] = []):
    """Returns a Python function that will be used by a Hugging Face Trainer to compute the specified 
    evaluation metrics (TrajMetric instances).

    Args:
        eval_metrics (list[TrajMetric], optional): List of TrajMetrics to be used for evaluating the model. Defaults to [].
    """

    # Define the Python function used to compute evaluation metrics
    def compute_metrics(eval_pred: EvalPrediction):
        """Returns a dictionary where the keys are the names of the TrajMetric instances,
        and the values are the corresponding computed values.

        Args:
            eval_pred (EvalPrediction): Evaluation output, including the predicted logits and the true labels for each input sample.
        """

        dict_metrics = {}
        for traj_metric in eval_metrics:
            metric_name = traj_metric.get_name()
            metric_result = traj_metric.compute(eval_pred)
            dict_metrics[metric_name] = metric_result
        return dict_metrics
        
    return compute_metrics