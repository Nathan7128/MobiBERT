"""
This Python module contains the implementation of the base class that represents a Hugging Face deep learning model used to process trajectory data.

This class serves as a template for all other classes of our library that represent more specific Hugging Face models
(Roberta for pre-training task, Deberta for fine-tuning, etc.).

This base class implements all the attributes and methods that serve as foundation for all
the models/subclasses (e.g., model saving, model name retrieval, etc.)
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

from abc import ABC, abstractmethod
from typing import Literal
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from datasets import Dataset
from transformers import PreTrainedModel
from transformers.trainer import Trainer
import torch

# Internal libraries

import sys
sys.path.append("mobiBERT")

from pipeline.hf_traj_datasets.datasets import HuggingFaceTrajDataset
from pipeline.traj_tokenizers.tokenizers import TrajTokenizer
from pipeline.traj_models.traj_mlflow import MLflowTracker

# --------------------------------------------------------- CLASS IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

class TrajModel(ABC):
    """Class that customizes a Hugging Face model used for trajectory data processing.

    This class allows the implementation of various model architectures with different hyperparameters (e.g., Roberta, Deberta, etc.).

    The models can be saved and retrieved (after training) from the library's data storage using their "model_name".  
    Before training your model, you should configure its architecture using the "config_model" method
    and its training parameters using "config_training".

    Optionally, you can also configure a Tracker for your model training monitoring, such as MLflow ("config_mlflow" method).

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to train the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
    """

    def __init__(self, model_name: str, dataset: HuggingFaceTrajDataset = None):
        # Name of the model used to store it in local data
        self._name = model_name
        # Path to the directory that contains the data related to this TrajModel
        self._model_dir: str = None
        # Path to the Hugging Face model
        self._hf_model_path: str = None
        # Path to the JSON file containing additional information about the TrajModel
        self._extra_info_path: str = None

        # Hugging Face model
        self._model: PreTrainedModel = None
        # Dictionary containing the main model parameters (selected by the developer)
        self._model_params: dict[str] = {}
        # Hugging Face Trainer used for model training
        self._trainer: Trainer = None
        # Boolean indicating whether the TrajModel has been trained or not
        self._is_trained = False

        # "Raw" HuggingFaceTrajDataset used to train the model
        self._dataset = dataset
        # Preprocessed Hugging Face dataset, ready to be used by the model
        self._preprocessed_dataset: Dataset = None
        
        # Tokenizer used to preprocess the dataset
        self._tokenizer: TrajTokenizer = None

        # MLflowTracker instance, used to track model training
        self._mlflow_tracker: MLflowTracker = None
        # Name of the configured tracker (if one is configured) which will be specified in the
        # Hugging Face TrainingArguments to enable the tracking of the model's training
        self._configured_tracker: str = None


    @staticmethod
    @abstractmethod
    def get_saved_models():
        pass


    @staticmethod
    @abstractmethod
    def load_from_file(model_dir: str) -> PreTrainedModel:
        """Loads and returns a TrajModel previously saved in the local data.

        This method also loads and assigns the TrajTokenizer previously assiocated to the saved TrajModel.  
        If this TrajTokenizer has been removed from the local files, this methods will raise an error.

        Args:
            model_dir (str): Path to the directory containing all the data related to the saved TrajModel.
        Returns:
            PreTrainedModel: TrajModel instance created from the data in the specified directory "model_dir".
        """
        
        pass


    @abstractmethod
    def save(self):
        """Save the model to the local models directory.

        The model is saved with the Hugging Face's "save_pretrained" function and can
        be retrieved using the specified "model_name" from the local storage.

        You cannot saved the TrajModel if:
            - The associated Hugging Face model has not been trained
            - A TrajModel with the same name already exists in the model storage directory

        Moreover, since all models are associated with and dependent on a tokenizer, this method automatically
        saves the TrajTokenizer associated with this TrajModel if it has not yet been saved in the local data.
        """

        if not(self._is_trained):
            raise ValueError("Error: You must train your model (using the 'train' method) before saving it!")
        if os.path.exists(self._model_dir):
            raise FileExistsError(f"Error: this TrajModel already exists: {self._model_dir}!\n"
                                  "Delete it from your files or load it using TrajModel.load_from_file method.")
        if not(os.path.exists(self._tokenizer.get_dir())):
            print("Caution: The TrajTokenizer associated with this TrajModel is not saved in your local data.\n"
                  "Saving it now...")
            self._tokenizer.save()
        
        # Save the Hugging Face model since all subclasses are associated with a model
        self._model.save_pretrained(save_directory=self._hf_model_path)


    @abstractmethod
    def preprocess_data(self):
        """Preprocesses the raw Hugging Face dataset (e.g., truncation, padding, etc.) associated with the TrajModel.

        This preprocessed dataset can then be used to train the model, evaluate its performance, etc.

        You cannot preprocess the raw dataset if:
            - It has been already preprocessed
            - No tokenizer is associated with the TrajModel (since the tokenization is at the core of preprocessing)
            - No HuggingFaceTrajDataset is associated with your TrajModel
        """

        if self._preprocessed_dataset is not None:
            raise ValueError("Error: The dataset is already preprocessed!")
        if self._tokenizer is None:
            raise ValueError("Error: There is no TrajTokenizer associated with this TrajModel!\n"
                             "Use the 'set_tokenizer' method.")
        if self._dataset is None:
            raise ValueError("Error: There is no HuggingFaceTrajDataset associated with this TrajModel!\n"
                             "Use the 'set_dataset' method.")

    
    @abstractmethod
    def config_model(self):
        if self._model is not None:
            raise ValueError("Error: The Hugging Face model associated with this TrajModel is already configured!")


    @abstractmethod
    def config_training(self):
        """Configures the training parameters and creates the Hugging Face Trainer used for model training.
        """

        if self._trainer is not None:
            raise ValueError("Error: The training of the Hugging Face model associated with this TrajModel is already configured!")
        if self._model is None:
            raise ValueError("Error: The Model must be instantied before configuring your training!\n"
                             "Use the 'config_model' method.")
        
    
    @abstractmethod
    def config_mlflow(self):
        if self._configured_tracker is not None:
            raise ValueError(f"Error: A Tracker has already been configured ({self._configured_tracker}) for the training of this model!")
        if self._trainer is None:
            raise ValueError("Error: You must configure the training of your model before configuring MLflow!\n"
                             "Use the 'config_training' method.")


    def train(self):
        """Trains the model using the specified tracking method (e.g., no tracking, MLflow, W&B, etc.).

        You cannot train your model if:
            - It has already been trained
            - CUDA is unvalaible on your device
            - You have not yet configured the training of your model
        """

        if self._is_trained:
            raise ValueError("Error: The Hugging Face model associated with this TrajModel has already been trained!")
        if not torch.cuda.is_available():
            raise Exception("Error: Cuda is not available!")
        if self._trainer is None:
            raise ValueError("Error: You must configure the training of your model!\n"
                             "Use the 'config_training' method.")
        torch.cuda.empty_cache()

        # MLflow tracking
        if self._configured_tracker == "mlflow":
            self._mlflow_tracker.train()

        # No tracking
        else:
            self._trainer.train()

        self._is_trained = True
        # Update the "_model" attribute with the PreTrainedModel resulting from the training performed by the Trainer instance
        self._model = self._trainer.model


    def get_name(self):
        """Returns the name of the TrajModel.
        """

        return self._name
    

    def get_dir(self):
        """Returns the path to the directory that contains all data related to the TrajModel.
        """

        return self._model_dir
    

    def get_hf_model_path(self):
        """Returns the path to the Hugging Face model associated with the TrajModel.
        """

        return self._hf_model_path
    

    def get_traj_tokenizer(self):
        """Returns the TrajTokenizer associated with your TrajModel.
        """

        if self._tokenizer is None:
            raise ValueError("Error: There is no TrajTokenizer associated with this TrajModel!\n"
                             "Use the 'set_tokenizer' method.")
        return self._tokenizer


    def get_model(self):
        """Returns the Hugging Face model associated with your TrajModel.
        """

        if self._model is None:
            raise ValueError("Error: The Hugging Face model associated with this TrajModel is not configured!\n"
                             "Use the 'config_model' method.")
        return self._model
    

    def get_model_params(self, choice: Literal["Main", "All"] = "Main") :
        """Returns a dictionary with either the main model parameters or all avalaible model parameters.

        Args:
            choice (str, optional): If "Main", returns only the main parameters of the
                model.  
                If "All" is specified, returns all available model parameters
                using the Hugging Face PreTrainedModel's `config` attribute.  
                Defaults to "Main".
        """

        if self._model_params is None :
            raise ValueError("Error: your model is not configured!\n"
                             "Use the 'config_model' method.")
        
        if choice == "Main":
            return self._model_params
        
        elif choice == "All":
            return self._model.config
        

    def set_dataset(self, dataset: HuggingFaceTrajDataset):
        """Associates a raw HuggingFaceTrajDataset with your TrajModel.
        """

        self._dataset = dataset
        self._preprocessed_dataset = None


    def set_tokenizer(self, tokenizer: TrajTokenizer):
        """Associates a TrajTokenizer with your TrajModel.
        """

        self._tokenizer = tokenizer