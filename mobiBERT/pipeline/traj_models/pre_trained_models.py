"""
This Python module contains the implementations of TrajModel subclasses that represent Hugging Face models used for pre-training tasks.

These models are designed to process trajectory data and should logically be fine-tuned after pre-training.
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import os
import json
from abc import abstractmethod

from transformers import (RobertaConfig, DebertaConfig, ModernBertConfig, RobertaForMaskedLM, DebertaForMaskedLM, ModernBertForMaskedLM,
                          DataCollatorForLanguageModeling, TrainingArguments, AutoModel)
from transformers.trainer import Trainer

# Internal libraries

import sys
sys.path.append("mobiBERT")

from pipeline.utils import get_module_classes
from pipeline.hf_traj_datasets.datasets import HuggingFaceTrajDataset
from pipeline.traj_tokenizers.tokenizers import TrajTokenizer
from pipeline.traj_models.traj_mlflow import MLflowTracker
from pipeline.traj_models.utils import pre_trained_models_dir, hf_model_dirname, extra_info_filename, update_special_tokens_ids
from pipeline.traj_models.traj_model import TrajModel

# --------------------------------------------------------- VARIABLE DEFINITIONS -----------------------------------------------------------------------------------------------------------------------

_model_checkpoints_dir = os.path.join(pre_trained_models_dir, "checkpoints")
"""Path to the directory that contains the model checkpoint data."""

# --------------------------------------------------------- CLASS IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

class TrajPreTrainedModel(TrajModel):
    """Class that inherits from TrajModel and represents a Hugging Face PreTrainedModel used for trajectory data processing.

    The pre-training technique used is the Masked Language Modeling, derived from self-supervised learning.

    Moreover, no evaluation metrics are computed during model training, only the training loss.  
    Therefore, the preprocessed dataset is not split into training, evaluation and test subsets, since the accuracy
    of a pre-trained model does not need to be assessed before it is fine-tuned later on.

    See the TrajModel documentation for additional information.

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to pre-train the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
        tokenizer (TrajTokenizer, optional): Tokenizer used to preprocess the data before it is used for model pre-training.  
            Defaults to None.
    """

    def __init__(self, model_name: str, dataset: HuggingFaceTrajDataset=None, tokenizer: TrajTokenizer=None):
        super().__init__(model_name=model_name, dataset=dataset)

        # Path to the directory that contains the data related to this TrajPreTrainedModel
        self._model_dir = os.path.join(pre_trained_models_dir, model_name)
        # Path to the Hugging Face PreTrainedModel
        self._hf_model_path : str = os.path.join(self._model_dir, hf_model_dirname)
        # Path to the JSON file containing additional information about the TrajPreTrainedModel
        self._extra_info_path = os.path.join(self._model_dir, extra_info_filename)
        
        # Tokenizer used to preprocess the dataset
        self._tokenizer = tokenizer


    @staticmethod
    def get_saved_models():
        """Returns a dictionary including the paths to the directories of all saved TrajPreTrainedModel.

        The keys are the name of the TrajPreTrainedModel, and the values are the absolute paths to their storage directories.
        """

        # List of the names of all saved TrajPreTrainedModel
        model_names_list = [model_name for model_name in os.listdir(pre_trained_models_dir)
                            if os.path.isdir(os.path.join(pre_trained_models_dir, model_name))]
        model_names_list.remove("checkpoints")
        # Dictionary that will contain the TrajPreTrainedModel names and the paths to their storage directories
        directories_dict = {model_name: os.path.join(pre_trained_models_dir, model_name)
                            for model_name in model_names_list}
        return directories_dict

        
    @staticmethod
    def load_from_file(model_dir):
        if not(os.path.exists(model_dir)):
            raise FileNotFoundError("Error: No TrajPreTrainedModel is stored here!")

        # Load the file that contains additional information
        extra_info_path = os.path.join(model_dir, extra_info_filename)
        with open(extra_info_path, "r") as f:
            extra_info = json.load(f)

        # Load the HuggingFaceTrajDataset previously associated with the saved TrajPreTrainedModel
        # if it exists in the local data
        dataset_dir = extra_info["dataset_dir"]
        if os.path.exists(dataset_dir):
            dataset = HuggingFaceTrajDataset.load_from_file(dataset_dir=dataset_dir)
        else :
            dataset = None

        # Load the TrajTokenizer previously assiocated with the saved TrajPreTrainedModel
        tokenizer_dir = extra_info["tokenizer_dir"]
        # Exception if it is no longer present in the local files
        if not(os.path.exists(tokenizer_dir)):
            raise FileNotFoundError("Error: The TrajTokenizer previously associated with this TrajPreTrainedModel has been deleted "
                                  f"or moved from your local data!\nExpected path: {tokenizer_dir}")
        tokenizer = TrajTokenizer.load_from_file(tokenizer_dir=tokenizer_dir)

        # Retrieve the class of the saved TrajPreTrainedModel
        model_classes = get_module_classes(sys.modules[__name__])
        model_class : type[TrajPreTrainedModel] = model_classes[extra_info["class"]]

        # Instantiate the new TrajPreTrainedModel from the information loaded from the saved TrajPreTrainedModel
        traj_model = model_class(model_name=extra_info["name"], dataset=dataset, tokenizer=tokenizer)
        traj_model._model_params = extra_info["model_params"]

        # Load the Hugging Face PreTrainedModel from the directory containing the saved TrajPreTrainedModel
        hf_model_path = os.path.join(model_dir, hf_model_dirname)
        traj_model._model = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_path)

        return traj_model


    def save(self):
        super().save()

        # Save the additional information related to the TrajPreTrainedModel
        model_info = {
            "name": self._name,        # Name of the TrajPreTrainedModel
            "class" : self.__class__.__name__,        # Name of the class of the TrajPreTrainedModel instance
            "model_params": self._model_params,        # _model_params attribute
            "dataset_dir": self._dataset.get_dir(),        # Path to the directory containing the associated HuggingFaceTrajDataset
            "tokenizer_dir": self._tokenizer.get_dir(),        # Path to the directory containing the associated TrajTokenizer
        }

        with open(self._extra_info_path, "w") as file:
            json.dump(obj=model_info, fp=file)


    @abstractmethod
    def config_training(self):
        super().config_training()
    
        # The entire initial dataset is used to pre-train the model, since it must be
        # preprocessed before being passed to the Hugging Face Trainer
        if self._preprocessed_dataset is None:
            raise ValueError("Error: You must preprocess the initial dataset before configuring your model's training!\n"
                             "Use the 'preprocess_data' method.")


    def config_mlflow(self, params_to_log: dict[str] = {}, track_loss=True, system_metrics=True):
        """Creates and configures an MLflow tracker to be used during model pre-training.

        Before configuring the MLflow tracker, you must set the training parameters of your model using the
        "config_training" method.

        See the MLflowTracker documentation for additional information.

        Args:
            params_to_log (dict[str], optional): Dictionnary of parameters (e.g., model hyperparameters) to be visualized in the MLflow UI.
                Use the "get_model_params" method to view the model parameters. Defaults to {}.
            track_loss (bool, optional): Whether to track the training loss during model training. Defaults to True.
            system_metrics (bool, optional): Whether to track system performance (e.g., CPU, GPU, memory, etc.)
                during model training. Defaults to True.
        """

        super().config_mlflow()

        # The only metric available to evaluate a TrajPreTrainedModel training, referred to as "loss" by Hugging Face
        loss = "loss" if track_loss else None

        # Create and set up the MLflowTracker instance with the configuration specified by the user
        mlflow_tracker = MLflowTracker(tracking_name=self._name, experiment_name="Pre-trained Models")
        mlflow_tracker.setup(trainer=self._trainer, params_to_log=params_to_log, system_metrics=system_metrics,
                             tracking_metrics=loss)
        self._mlflow_tracker = mlflow_tracker

        self._configured_tracker = "mlflow"



class DebertaTrajPreTrained(TrajPreTrainedModel) :
    """Class that inherits from TrajPreTrainedModel and enables the pre-training of the Deberta model.

    See the TrajPreTrainedModel documentation for additional information.

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to pre-train the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
        tokenizer (TrajTokenizer, optional): Tokenizer used to preprocess the data before it is used for model pre-training.  
            Defaults to None.
    """

    def __init__(self, model_name: str, dataset=None, tokenizer=None):
        super().__init__(model_name, dataset, tokenizer)

    
    def preprocess_data(self):
        super().preprocess_data()

        # The preprocessing only consists of tokenizing the dataset
        self._preprocessed_dataset = self._tokenizer.tokenize_dataset(dataset=self._dataset.get_hf_dataset())
        self._preprocessed_dataset = self._preprocessed_dataset.remove_columns(column_names=["text", "label", "timestamps"])


    def config_model(self, num_attention_heads=12, num_hidden_layers=12, hidden_size=768, intermediate_size=3072):
        """Configures the architecture of the Deberta model by instantiating a new Hugging Face model used for Masked Language Modeling.

        This method also assigns to this TrajPreTrainedModel the parameters specified as arguments.

        Args:
            num_attention_heads (int, optional): Number of heads used in the MHA layers of the encoder. Defaults to 12.
            num_hidden_layers (int, optional): Number of attention blocks in the encoder (MHA + FFN). Defaults to 12.
            hidden_size (int, optional): Dimensionality of the encoder layers. Defaults to 768.
            intermediate_size (int, optional): Dimension of the FFN. Defaults to 3072.
        """

        super().config_model()

        # Create a dictionary containing the function's parameters (i.e., the model parameters)
        model_params = {param_name: param_value for param_name, param_value in locals().copy().items()
                        if param_name not in ["self", "__class__"]} # "self" and "__class__" are automatically included in the locals().copy() dictionary
        # Add this dictionary to the main model parameters dictionary
        self._model_params.update(model_params)

        # Create a Hugging Face model configuration instance
        config = DebertaConfig(
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_position_embeddings=self._tokenizer.get_max_position_embeddings(),
            vocab_size=self._tokenizer.get_vocab_size(),
        )
        # Update special token IDs
        config = update_special_tokens_ids(tokenizer=self._tokenizer, config=config)
        
        # Create the PreTrainedModel instance
        self._model = DebertaForMaskedLM(config=config)
        # Add the number of model architecture parameters to _model_params
        self._model_params["num_params"] = self._model.num_parameters()
    

    def config_training(self, mlm_probability=0.15, nb_epochs=2, batch_size=16):
        """Configures the training parameters, automatically preprocesses the dataset ("preprocess_data" method)
        and creates the Hugging Face Trainer used for model training.

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.15.
            nb_epochs (int, optional): Number of training epochs. Defaults to 2.
            batch_size (int, optional): Size of input batches. Defaults to 32.
        """

        super().config_training()

        # Data Collator instance for Masked Language Modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer.get_hf_tokenizer(), mlm=True,
                                                        mlm_probability=mlm_probability)
        
        # Configure the traning arguments used to create the Trainer instance
        training_args = TrainingArguments(
            output_dir=os.path.join(_model_checkpoints_dir, self._name),      # Path to the directory that will contain the model training checkpoints
            overwrite_output_dir=True,
            num_train_epochs=nb_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=2000,
            save_total_limit=20,
            prediction_loss_only=True,
            report_to=self._configured_tracker,
        )

        # Create the model Trainer instance
        self._trainer = Trainer(model=self._model,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=self._preprocessed_dataset)



class RobertaTrajPreTrained(TrajPreTrainedModel):
    """Class that inherits from TrajPreTrainedModel and enables the pre-training of the Roberta model.

    See the TrajPreTrainedModel documentation for additional information.

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to pre-train the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
        tokenizer (TrajTokenizer, optional): Tokenizer used to preprocess the data before it is used for model pre-training.  
            Defaults to None.
    """

    def __init__(self, model_name: str, dataset=None, tokenizer=None):
        super().__init__(model_name, dataset, tokenizer)

    
    def preprocess_data(self):
        super().preprocess_data()

        # The preprocessing only consists of tokenizing the dataset
        self._preprocessed_dataset = self._tokenizer.tokenize_dataset(dataset=self._dataset.get_hf_dataset())
        self._preprocessed_dataset = self._preprocessed_dataset.remove_columns(column_names=["text", "label", "timestamps"])


    def config_model(self, num_attention_heads=12, num_hidden_layers=12, hidden_size=768, intermediate_size=3072):
        """Configures the architecture of the Roberta model by instantiating a new Hugging Face model used for Masked Language Modeling.

        This method also assigns to this TrajPreTrainedModel the parameters specified as arguments.

        Args:
            num_attention_heads (int, optional): Number of heads used in the MHA layers of the encoder. Defaults to 12.
            num_hidden_layers (int, optional): Number of attention blocks in the encoder (MHA + FFN). Defaults to 12.
            hidden_size (int, optional): Dimensionality of the encoder layers. Defaults to 768.
            intermediate_size (int, optional): Dimension of the FFN. Defaults to 3072.
        """

        super().config_model()

        # Create a dictionary containing the function's parameters (i.e., the model parameters)
        model_params = {param_name: param_value for param_name, param_value in locals().copy().items()
                        if param_name not in ["self", "__class__"]} # "self" and "__class__" are automatically included in the locals().copy() dictionary
        # Add this dictionary to the main model parameters dictionary
        self._model_params.update(model_params)

        # Create a Hugging Face model configuration instance
        config = RobertaConfig(
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_position_embeddings=self._tokenizer.get_max_position_embeddings(),
            vocab_size=self._tokenizer.get_vocab_size(),
        )
        # Update special token IDs
        config = update_special_tokens_ids(tokenizer=self._tokenizer, config=config)

        # Create the PreTrainedModel instance
        self._model = RobertaForMaskedLM(config=config)
        # Add the number of model architecture parameters to _model_params
        self._model_params["num_params"] = self._model.num_parameters()

    
    def config_training(self, mlm_probability=0.15, nb_epochs=2, batch_size=32):
        """Configures the training parameters, automatically preprocesses the dataset ("preprocess_data" method)
        and creates the Hugging Face Trainer used for model training.

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.15.
            nb_epochs (int, optional): Number of training epochs. Defaults to 2.
            batch_size (int, optional): Size of input batches. Defaults to 32.
        """
        
        super().config_training()

        # Data Collator instance for Masked Language Modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer.get_hf_tokenizer(), mlm=True,
                                                        mlm_probability=mlm_probability)
        
        # Configure the traning arguments used to create the Trainer instance
        training_args = TrainingArguments(
            output_dir=os.path.join(_model_checkpoints_dir, self._name),      # Path to the directory that will contain the model training checkpoints
            overwrite_output_dir=True,
            num_train_epochs=nb_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=2000,
            save_total_limit=20,
            prediction_loss_only=True,
            fp16=True,
            report_to=self._configured_tracker,
        )

        # Create the model Trainer instance
        self._trainer = Trainer(model=self._model,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=self._preprocessed_dataset)
                


class ModernBertTrajPreTrained(TrajPreTrainedModel) :
    """Class that inherits from TrajPreTrainedModel and enables the pre-training of the Roberta model.

    See the TrajPreTrainedModel documentation for additional information.

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to pre-train the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
        tokenizer (TrajTokenizer, optional): Tokenizer used to preprocess the data before it is used for model pre-training.  
            Defaults to None.
    """

    def __init__(self, model_name: str, dataset=None, tokenizer=None):
        super().__init__(model_name, dataset, tokenizer)
    

    def preprocess_data(self):
        super().preprocess_data()

        # The preprocessing only consists of tokenizing the dataset
        self._preprocessed_dataset = self._tokenizer.tokenize_dataset(dataset=self._dataset.get_hf_dataset())
        self._preprocessed_dataset = self._preprocessed_dataset.remove_columns(column_names=["text", "label", "timestamps"])

    
    def config_model(self, num_attention_heads=12, num_hidden_layers=22, hidden_size=768, intermediate_size=1152):
        """Configures the architecture of the ModernBert model by instantiating a new Hugging Face model used for Masked Language Modeling.

        This method also assigns to this TrajPreTrainedModel the parameters specified as arguments.

        Args:
            num_attention_heads (int, optional): Number of heads used in the MHA layers of the encoder. Defaults to 12.
            num_hidden_layers (int, optional): Number of attention blocks in the encoder (MHA + FFN). Defaults to 12.
            hidden_size (int, optional): Dimensionality of the encoder layers. Defaults to 768.
            intermediate_size (int, optional): Dimension of the FFN. Defaults to 1152.
        """

        super().config_model()
        
        # Create a dictionary containing the function's parameters (i.e., the model parameters)
        model_params = {param_name: param_value for param_name, param_value in locals().copy().items()
                        if param_name not in ["self", "__class__"]} # "self" and "__class__" are automatically included in the locals().copy() dictionary
        # Add this dictionary to the main model parameters dictionary
        self._model_params.update(model_params)

        # Create a Hugging Face model configuration instance
        config = ModernBertConfig(
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_position_embeddings=self._tokenizer.get_max_position_embeddings(),
            vocab_size=self._tokenizer.get_vocab_size(),
        )
        # Update special token IDs
        config = update_special_tokens_ids(tokenizer=self._tokenizer, config=config)
        
        # Create the PreTrainedModel instance
        self._model = ModernBertForMaskedLM(config=config)
        # Add the number of model architecture parameters to _model_params
        self._model_params["num_params"] = self._model.num_parameters()


    def config_training(self, mlm_probability=0.30, nb_epochs=2, batch_size=16):
        """Configures the training parameters, automatically preprocesses the dataset ("preprocess_data" method)
        and creates the Hugging Face Trainer used for model training.

        Args:
            mlm_probability (float, optional): The probability with which to (randomly) mask tokens in the input. Defaults to 0.30.
            nb_epochs (int, optional): Number of training epochs. Defaults to 2.
            batch_size (int, optional): Size of input batches. Defaults to 32.
        """
        
        super().config_training()

        # Data Collator instance for Masked Language Modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer.get_hf_tokenizer(), mlm=True,
                                                        mlm_probability=mlm_probability)
        
        # Configure the traning arguments used to create the Trainer instance
        training_args = TrainingArguments(
            output_dir=os.path.join(_model_checkpoints_dir, self._name),      # Path to the directory that will contain the model training checkpoints
            overwrite_output_dir=True,
            num_train_epochs=nb_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=2000,
            save_total_limit=20,
            prediction_loss_only=True,
            report_to=self._configured_tracker,
        )

        # Create the model Trainer instance
        self._trainer = Trainer(model=self._model,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=self._preprocessed_dataset)