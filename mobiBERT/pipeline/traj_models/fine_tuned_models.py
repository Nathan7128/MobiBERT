"""
This Python module contains the implementations of the classes that represent Hugging Face models used for fine-tuning task.
These models are implemented to process trajectory data.
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import os
import json
from abc import abstractmethod

from transformers import (AutoModelForSequenceClassification, TrainingArguments)
from transformers.trainer import Trainer
from datasets import Dataset

# Internal libraries

import sys
sys.path.append("mobiBERT")

from pipeline.utils import get_module_classes
from pipeline.hf_traj_datasets.datasets import HuggingFaceTrajDataset
from pipeline.traj_tokenizers.tokenizers import TrajTokenizer
from pipeline.traj_models.traj_model import TrajModel
from pipeline.traj_models.pre_trained_models import TrajPreTrainedModel
from pipeline.traj_models.utils import fine_tuned_models_dir, hf_model_dirname, extra_info_filename, get_compute_metrics_fn
from pipeline.traj_models.traj_mlflow import MLflowTracker
from pipeline.traj_models.traj_metrics import TrajMetric

# --------------------------------------------------------- VARIABLE DEFINITIONS -----------------------------------------------------------------------------------------------------------------------

_model_checkpoints_path = os.path.join(fine_tuned_models_dir, "checkpoints")
"""Path to the directory that contains the model checkpoint data."""

# --------------------------------------------------------- CLASS IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

class TrajFineTunedModel(TrajModel):
    """Class that inherits from TrajModel and represents Hugging Face models to be fine-tuned.

    These models are used for trajectory data processing.  
    Before fine-tuning, they must be configured from a pre-trained model (i.e., a TrajPreTrainedModel) using the "config_model" method.

    The tokenizer associated with a specific TrajFineTunedModel is the same as the one used by the specified TrajPreTrainedModel.  
    Similarly, the saved model parameters (the "model_params" attribute) of the TrajFineTunedModel are automatically loaded
    based on the specified TrajPreTrainedModel.  

    Moreover, this class allows you to split the HuggingFaceTrajDataset passed as an argument into 3 datasets (train, evaluation and test)
    for training and evaluation purposes.  
    Unlike pre-trained models, it is useful to evaluate the performance of a fine-tuned model using various metrics
    during training.  
    These metrics are computed on the evaluation dataset.

    See the TrajModel documentation for additional information.

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to fine-tune the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
    """

    def __init__(self, model_name: str, dataset: HuggingFaceTrajDataset = None):
        super().__init__(model_name = model_name, dataset = dataset)

        # Path to the directory that contains the data related to this TrajPreTrainedModel
        self._model_dir = os.path.join(fine_tuned_models_dir, model_name)
        # Path to the Hugging Face PreTrainedModel
        self._hf_model_path: str = os.path.join(self._model_dir, hf_model_dirname)
        # Path to the JSON file containing additional information about the TrajPreTrainedModel
        self._extra_info_path = os.path.join(self._model_dir, extra_info_filename)

        # HuggingFaceTrajDataset use to fine-tune the pre-trained model
        self._dataset = dataset
        # Dataset used to train the model
        self._train_dataset: Dataset = None
        # Dataset used for model evaluation
        self._eval_dataset: Dataset = None
        # Dataset used to test the model
        self._test_dataset: Dataset = None

        # List of metrics (TrajMetric instances) used for model evaluation during fine-tuning
        self._eval_metrics: list[TrajMetric] = []


    @staticmethod
    def get_saved_models():
        """Returns a dictionary including the paths to the directories of all saved TrajFineTunedModel.

        The keys are the name of the TrajFineTunedModel, and the values are the absolute paths to their storage directories.
        """

        # List of the names of all saved TrajFineTunedModel
        model_names_list = [model_name for model_name in os.listdir(fine_tuned_models_dir)
                            if os.path.isdir(os.path.join(fine_tuned_models_dir, model_name))]
        model_names_list.remove("checkpoints")
        # Dictionary that will contain the TrajFineTunedModel names and the paths to their storage directories
        directories_dict = {model_name: os.path.join(fine_tuned_models_dir, model_name)
                            for model_name in model_names_list}
        return directories_dict

        
    @staticmethod
    def load_from_file(model_dir):
        if not(os.path.exists(model_dir)):
            raise FileNotFoundError("Error: No TrajFineTunedModel is stored here!")
    
        # Load the file that contains additional information
        extra_info_path = os.path.join(model_dir, extra_info_filename)
        with open(extra_info_path, "r") as f:
            extra_info = json.load(f)

        # Load the HuggingFaceTrajDataset previously associated with the saved TrajFineTunedModel
        # if it exists in the local data
        dataset_dir = extra_info["dataset_dir"]
        if os.path.exists(dataset_dir):
            dataset = HuggingFaceTrajDataset.load_from_file(dataset_dir=dataset_dir)
        else:
            dataset = None

        # Retrieve the class of the saved TrajFineTunedModel
        model_classes = get_module_classes(sys.modules[__name__])
        model_class: type[TrajFineTunedModel] = model_classes[extra_info["class"]]
        # Instantiate the new TrajFineTunedModel based on the loaded information
        traj_model = model_class(model_name=extra_info["name"], dataset=dataset)
        # Load the fine-tuned Hugging Face model and assign it to the new TrajFineTunedModel
        hf_model_path = os.path.join(model_dir, hf_model_dirname)
        traj_model._model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model_name_or_path=hf_model_path)
        traj_model._model_params = extra_info["model_params"]

        # Load the TrajPreTrainedModel that was previously associated with the saved TrajFineTunedModel
        # Since every saved TrajFineTunedModel has already been fine-tuned, assigning a TrajPreTrainedModel is not mandatory
        pre_trained_model_dir = extra_info["pre_trained_model_dir"]
        if os.path.exists(pre_trained_model_dir):
            pre_trained_model = TrajPreTrainedModel.load_from_file(model_dir = pre_trained_model_dir)
        else :
            pre_trained_model = None
        traj_model._pre_trained_model = pre_trained_model

        # Load the TrajTokenizer previously assiocated with the saved TrajFineTunedModel
        tokenizer_dir = extra_info["tokenizer_dir"]
        # Exception if it is no longer present in the local files
        if not(os.path.exists(tokenizer_dir)):
            raise FileNotFoundError("Error: The TrajTokenizer associated with this TrajFineTunedModel has been deleted "
                                  f"or moved from your local data!\nExpected path: {tokenizer_dir}")
        tokenizer = TrajTokenizer.load_from_file(tokenizer_dir=tokenizer_dir)
        traj_model._tokenizer = tokenizer

        return traj_model
        

    def save(self):
        super().save()

        # Save the additional information related to the TrajFineTunedModel
        model_info = {
            "name": self._name,        # Name of the TrajFineTunedModel
            "class" : self.__class__.__name__,        # Name of the class of the TrajFineTunedModel instance
            "model_params": self._model_params,        # _model_params attribute
            "dataset_dir": self._dataset.get_dir(),        # Path to the directory containing the associated HuggingFaceTrajDataset
            "tokenizer_dir": self._tokenizer.get_dir(),        # Path to the directory containing the associated TrajTokenizer
        }

        with open(self._extra_info_path, "w") as file:
            json.dump(obj=model_info, fp=file)


    def config_model(self, pre_trained_model: TrajPreTrainedModel):
        """Configures the initial fine-tuned Hugging Face model by using a pre-trained model as a base (the "pre_trained_model" argument).

        During this configuration, the method also assigns to the TrajFineTunedModel the same saved model parameters and TrajTokenizer as
        those used by the specified TrajPreTrainedModel.

        The Hugging Face model configured here is specific to classification tasks (using the
        "AutoModelForSequenceClassification" class from Hugging Face).

        Args:
            pre_trained_model (TrajPreTrainedModel): Pre-trained model that will serve as the base for the fine-tuned model.
        """

        super().config_model()

        # Retrieve and assign to the TrajFineTunedModel the saved model parameters of the TrajPreTrainedModel
        self._model_params = pre_trained_model.get_model_params(choice="Main")
        # Retrieve and assign to the TrajFineTunedModel the TrajTokenizer previously associated with the TrajPreTrainedModel
        self._tokenizer = pre_trained_model.get_traj_tokenizer()

        # Instantiate the new fine-tuned Hugging Face model used for classifications task
        self._model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pre_trained_model.get_hf_model_path(),
            num_labels=self._dataset.get_nb_users(),      # Number of unique user IDs used as classification labels
            id2label=self._dataset.get_ids_to_labels(),       # Dictionnary that maps index to user ID
            label2id=self._dataset.get_labels_to_ids(),       # Dictionnary that maps user ID to index
            output_attentions=False
        )

    
    @abstractmethod
    def config_training(self):
        super().config_training()
    
        # Exception if the dataset has not been split before configuring the model training, since
        # we pass the training and evaluation subsets to the Hugging Face Trainer
        if self._train_dataset is None or self._eval_dataset is None or self._test_dataset is None:
            raise ValueError("Error: You must split your dataset into training, evaluation and test sets "
                             "before configuring the training of your TrajFineTunedModel!\n"
                             "Use the 'split_dataset' method.")


    def split_dataset(self, train_size=0.6, eval_size=0.2, test_size=0.2):
        """Splits the preprocessed version of the initial HuggingFaceTrajDataset (provided to the constructor)
        into 3 subsets: training, evaluation and test.

        Args:
            train_size (float, optional): Proportion of the dataset to include in the training split. Defaults to 0.6.
            eval_size (float, optional): Proportion of the dataset to include in the evaluation split. Defaults to 0.2.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        """

        # Error if the proportions passed as arguments are inconsistent
        if train_size + eval_size + test_size != 1:
            raise ValueError("Error: the sum of the train, eval, and test sizes must be equal to 1!")
        
        # The initial dataset must be preprocessed before being split
        if self._preprocessed_dataset is None :
            raise ValueError("Error: You must preprocess your initial raw dataset before splitting it into training, evaluation, and test subsets!\n"
                             "Use the 'preprocess_data' method.")
        
        splitted_dataset = self._preprocessed_dataset.train_test_split(train_size=train_size)
        self._train_dataset = splitted_dataset["train"]

        new_eval_size = eval_size/(eval_size + test_size)
        splitted_test_dataset = splitted_dataset["test"].train_test_split(train_size=new_eval_size)
        self._eval_dataset = splitted_test_dataset["train"]
        self._test_dataset = splitted_test_dataset["test"]


    def config_mlflow(self, params_to_log: dict[str] = {}, track_train_loss = True, track_eval_loss = True, system_metrics=True):
        """Configures and creates an MLflow tracker to be used during model fine-tuning.

        Before configuring the MLflow tracker, you must set the training parameters of your model using the
        "config_training" method.

        See the MLflowTracker documentation for additional information.

        Args:
            params_to_log (dict[str], optional): Dictionnary of parameters (e.g., model hyperparameters) to be visualized in the MLflow UI.
                Use the "get_model_params" method to view the model parameters. Defaults to {}.
            track_train_loss (bool, optional): Whether to track the training loss during model training. Defaults to True.
            track_eval_loss (bool, optional): Whether to track the evaluation loss during model training. Defaults to True.
            system_metrics (bool, optional): Whether to track system performance (e.g., CPU, GPU, memory, etc.)
                during model training. Defaults to True.
        """

        super().config_mlflow()

        # Whether to track training and evaluation losses, referred to as "loss" and "eval_loss" by Hugging Face
        training_loss = "loss" if track_train_loss else None
        eval_loss = "eval_loss" if track_eval_loss else None

        # List of names of all specified evaluation metrics (TrajMetric instances) to be tracked during the model's fine-tuning
        # All the evaluation metrics are automatically prefixed with "eval_" by Hugging Face
        tracking_metrics = [training_loss, eval_loss] + ["eval_" + metric.get_name() for metric in self._eval_metrics]
        tracking_metrics = [metric for metric in tracking_metrics if metric is not None]

        # Create and set up the MLflowTracker instance with the configuration specified by the user
        mlflow_tracker = MLflowTracker(tracking_name=self._name, experiment_name="Fine-tuned Models")
        mlflow_tracker.setup(trainer=self._trainer, params_to_log=params_to_log, system_metrics=system_metrics,
                             tracking_metrics=tracking_metrics)
        self._mlflow_tracker = mlflow_tracker

        self._configured_tracker = "mlflow"


    def evaluate(self, eval_metrics: list[TrajMetric] = []):
        """Evaluates the model on the test dataset using a specified list of TrajMetrics.

        Args:
            eval_metrics (list[TrajMetric], optional): List of TrajMetrics to use for evaluating the model. Defaults to [].
        """

        if self._is_trained is False:
            raise ValueError("Error: You must train your model (using the 'train' method) before evaluating it!")
        elif len(eval_metrics) == 0:
            raise ValueError("Error: You must provide at least one TrajMetric to evaluate your model!")
        elif self._test_dataset is None:
            raise ValueError("Error: No test dataset is associated with this TrajFineTunedModel!\n"
                             "Use the 'split_dataset' method.")

        # Create a new Trainer based on the trained model and the test dataset.
        # This is necessary if this TrajFineTunedModel instance was loaded from files,
        # it is not assiociated with a Trainer instance
        eval_trainer = Trainer(model=self._model,
                               eval_dataset=self._test_dataset,
                               processing_class = self._tokenizer.get_hf_tokenizer(),
                               compute_metrics = get_compute_metrics_fn(eval_metrics),)
        return eval_trainer.evaluate()
        


class DebertaTrajFineTuned(TrajFineTunedModel):
    """Class that inherits from TrajFineTunedModel and allows fine-tuning of a pre-trained model based on the Deberta architecture.

    See the TrajFineTunedModel documentation for additional information.

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to fine-tune the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
    """

    def __init__(self, model_name: str, dataset: HuggingFaceTrajDataset):
        super().__init__(model_name, dataset)


    def preprocess_data(self):
        super().preprocess_data()

        # Tokenize the initial dataset
        self._preprocessed_dataset = self._tokenizer.tokenize_dataset(dataset=self._dataset.get_hf_dataset())

        labels_to_ids = self._dataset.get_labels_to_ids()
        # Replace the "label" column, which contains the target values for classification task.
        # Originally, this column contained user IDs. We replace them with a range of integer indices.
        self._preprocessed_dataset = self._preprocessed_dataset.map(lambda x: {'label': labels_to_ids[x['label']]})
        

    def config_training(self, eval_metrics: list[TrajMetric] = [], learning_rate=2e-5, nb_epochs=8,
                        weight_decay=0.01, train_batch_size=32, eval_batch_size=16):
        """Configures the training parameters and assigns to the Hugging Face trainer a list of specified TrajMetrics
        to be used in order to evaluate the model's performance throughout training.

        Args:
            eval_metrics (list[TrajMetric], optional): List of TrajMetrics to be used to evaluate
                the model throughout training. Defaults to [].
            learning_rate (_type_, optional): The initial learning rate for Gradient descent. Defaults to 2e-5.
            nb_epochs (int, optional): Number of training epochs. Defaults to 8.
            weight_decay (float, optional): L2 regularization to apply to all layers except all bias and normalization layers. Defaults to 0.01.
            train_batch_size (int, optional): The batch size per device accelerator core/CPU for training. Defaults to 64.
            eval_batch_size (int, optional): The batch size per device accelerator core/CPU for evaluation. Defaults to 16.
        """

        super().config_training()

        # Set the specified TrajMetrics to be used to evaluate the model throughout training
        self._eval_metrics = eval_metrics

        # Add the specified learning rate to the main model parameters dictionary
        self._model_params["learning_rate"] = learning_rate
        
        # Configure the traning arguments used to create the Trainer instance
        training_args = TrainingArguments(
            output_dir = os.path.join(_model_checkpoints_path, self._name),
            learning_rate = learning_rate,
            per_device_train_batch_size = train_batch_size,
            per_device_eval_batch_size = eval_batch_size,
            num_train_epochs = nb_epochs,
            weight_decay = weight_decay,
            logging_strategy="steps",
            logging_steps = 100,
            eval_strategy = "steps",
            eval_steps = 100,
            save_strategy = "steps",
            load_best_model_at_end = True,
            report_to=self._configured_tracker,
        )

        # Create the model Trainer instance
        self._trainer = Trainer(
            model = self._model,
            args = training_args,
            train_dataset = self._train_dataset,
            eval_dataset = self._eval_dataset,
            processing_class = self._tokenizer.get_hf_tokenizer(),
            compute_metrics = get_compute_metrics_fn(eval_metrics) if len(self._eval_metrics) > 0 else None,
        )



class RobertaTrajFineTuned(TrajFineTunedModel):
    """Class that inherits from TrajFineTunedModel and allows fine-tuning of a pre-trained model based on the Roberta architecture.

    See the TrajFineTunedModel documentation for additional information.

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to fine-tune the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
    """

    def __init__(self, model_name: str, dataset: HuggingFaceTrajDataset):
        super().__init__(model_name, dataset)


    def preprocess_data(self):
        super().preprocess_data()

        # Tokenize the initial dataset
        self._preprocessed_dataset = self._tokenizer.tokenize_dataset(dataset=self._dataset.get_hf_dataset())

        labels_to_ids = self._dataset.get_labels_to_ids()
        # Replace the "label" column, which contains the target values for classification task.
        # Originally, this column contained user IDs. We replace them with a range of integer indices.
        self._preprocessed_dataset = self._preprocessed_dataset.map(lambda x: {'label': labels_to_ids[x['label']]})
        

    def config_training(self, eval_metrics: list[TrajMetric] = [], learning_rate=2e-5, nb_epochs=8,
                        weight_decay=0.01, train_batch_size=64, eval_batch_size=16):
        """Configures the training parameters and assigns to the Hugging Face trainer a list of specified TrajMetrics
        to be used in order to evaluate the model's performance throughout training.

        Args:
            eval_metrics (list[TrajMetric], optional): List of TrajMetrics to be used to evaluate
                the model throughout training. Defaults to [].
            learning_rate (_type_, optional): The initial learning rate for Gradient descent. Defaults to 2e-5.
            nb_epochs (int, optional): Number of training epochs. Defaults to 8.
            weight_decay (float, optional): L2 regularization to apply to all layers except all bias and normalization layers. Defaults to 0.01.
            train_batch_size (int, optional): The batch size per device accelerator core/CPU for training. Defaults to 64.
            eval_batch_size (int, optional): The batch size per device accelerator core/CPU for evaluation. Defaults to 16.
        """

        super().config_training()

        # Set the specified TrajMetrics to be used to evaluate the model throughout training
        self._eval_metrics = eval_metrics

        # Add the specified learning rate to the main model parameters dictionary
        self._model_params["learning_rate"] = learning_rate
        
        # Configure the traning arguments used to create the Trainer instance
        training_args = TrainingArguments(
            output_dir = os.path.join(_model_checkpoints_path, self._name),
            learning_rate = learning_rate,
            per_device_train_batch_size = train_batch_size,
            per_device_eval_batch_size = eval_batch_size,
            num_train_epochs = nb_epochs,
            weight_decay = weight_decay,
            logging_strategy="steps",
            logging_steps = 100,
            eval_strategy = "steps",
            eval_steps = 100,
            save_strategy = "steps",
            load_best_model_at_end = True,
            fp16 = True,
            report_to=self._configured_tracker,
        )

        # Create the model Trainer instance
        self._trainer = Trainer(
            model = self._model,
            args = training_args,
            train_dataset = self._train_dataset,
            eval_dataset = self._eval_dataset,
            processing_class = self._tokenizer.get_hf_tokenizer(),
            compute_metrics = get_compute_metrics_fn(eval_metrics) if len(self._eval_metrics) > 0 else None,
        )



class ModernBertTrajFineTuned(TrajFineTunedModel):
    """Class that inherits from TrajFineTunedModel and allows fine-tuning of a pre-trained model based on the ModernBERT architecture.

    See the TrajFineTunedModel documentation for additional information.

    Args:
        model_name (str): Name used to store the model in the library's local data storage.
        dataset (HuggingFaceTrajDataset, optional): Dataset used to fine-tune the model.  
            The input data should not be preprocessed (padded, truncated, etc.).  
            It must include at least a "text" column containing the text sequences used for training. Defaults to None.
    """

    def __init__(self, model_name: str, dataset: HuggingFaceTrajDataset):
        super().__init__(model_name, dataset)


    def preprocess_data(self):
        super().preprocess_data()

        # Tokenize the initial dataset
        self._preprocessed_dataset = self._tokenizer.tokenize_dataset(dataset=self._dataset.get_hf_dataset())

        labels_to_ids = self._dataset.get_labels_to_ids()
        # Replace the "label" column, which contains the target values for classification task.
        # Originally, this column contained user IDs. We replace them with a range of integer indices.
        self._preprocessed_dataset = self._preprocessed_dataset.map(lambda x: {'label': labels_to_ids[x['label']]})
        

    def config_training(self, eval_metrics: list[TrajMetric] = [], learning_rate=2e-5, nb_epochs=8,
                        weight_decay=0.01, train_batch_size=32, eval_batch_size=16):
        """Configures the training parameters and assigns to the Hugging Face trainer a list of specified TrajMetrics
        to be used in order to evaluate the model's performance throughout training.

        Args:
            eval_metrics (list[TrajMetric], optional): List of TrajMetrics to be used to evaluate
                the model throughout training. Defaults to [].
            learning_rate (_type_, optional): The initial learning rate for Gradient descent. Defaults to 2e-5.
            nb_epochs (int, optional): Number of training epochs. Defaults to 8.
            weight_decay (float, optional): L2 regularization to apply to all layers except all bias and normalization layers. Defaults to 0.01.
            train_batch_size (int, optional): The batch size per device accelerator core/CPU for training. Defaults to 64.
            eval_batch_size (int, optional): The batch size per device accelerator core/CPU for evaluation. Defaults to 16.
        """

        super().config_training()

        # Set the specified TrajMetrics to be used to evaluate the model throughout training
        self._eval_metrics = eval_metrics

        # Add the specified learning rate to the main model parameters dictionary
        self._model_params["learning_rate"] = learning_rate
        
        # Configure the traning arguments used to create the Trainer instance
        training_args = TrainingArguments(
            output_dir = os.path.join(_model_checkpoints_path, self._name),
            learning_rate = learning_rate,
            per_device_train_batch_size = train_batch_size,
            per_device_eval_batch_size = eval_batch_size,
            num_train_epochs = nb_epochs,
            weight_decay = weight_decay,
            logging_strategy="steps",
            logging_steps = 100,
            eval_strategy = "steps",
            eval_steps = 100,
            save_strategy = "steps",
            load_best_model_at_end = True,
            report_to=self._configured_tracker,
        )

        # Create the model Trainer instance
        self._trainer = Trainer(
            model = self._model,
            args = training_args,
            train_dataset = self._train_dataset,
            eval_dataset = self._eval_dataset,
            processing_class = self._tokenizer.get_hf_tokenizer(),
            compute_metrics = get_compute_metrics_fn(eval_metrics) if len(self._eval_metrics) > 0 else None,
        )