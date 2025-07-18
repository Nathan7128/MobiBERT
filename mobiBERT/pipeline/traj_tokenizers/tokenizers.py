"""
This Python module contains the implementation of the custom Hugging Face tokenizers used to process trajectory data (spatio-temporal sequences).
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import os
import json
from abc import ABC, abstractmethod
from functools import partial
from typing import Any

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import AutoTokenizer, RobertaTokenizerFast, DebertaTokenizerFast, PreTrainedTokenizerBase
from datasets import Dataset

# Internal libraries

import sys
sys.path.append("mobiBERT")

from pipeline.utils import get_module_classes
from pipeline.traj_tokenizers.utils import batch_iterator, tokenizers_dir, hf_tokenizer_dirname, extra_info_filename
from pipeline.hf_traj_datasets.datasets import HuggingFaceTrajDataset

# --------------------------------------------------------- CLASS IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

class TrajTokenizer(ABC):
    """Class that customizes and trains a Hugging Face Tokenizer associated with a trajectory dataset.

    To get the final Hugging Face tokenizer associated with a TrajTokenizer,
    you must initialize and train it (using the "initialize" and "train" methods).

    You must subclass this class to create a specific tokenizer (e.g., a tokenizer
    used to process spatio-temporal sequences for Roberta model).

    Each tokenizer can be saved in a directory within your local files, which have
    the same name as the tokenizer's "tokenizer_name".  
    However, a TrajTokenizer that has not been initialized and trained
    cannot be saved to your local data.

    Args:
        tokenizer_name (str): Name of the tokenizer used for storage.
        dataset (HuggingFaceTrajDataset, optional): Trajectory dataset used to train the tokenizer. Defaults to None.
    """

    def __init__(self, tokenizer_name: str, dataset: HuggingFaceTrajDataset = None):
        if tokenizer_name is None:
            raise ValueError("Error: You must specify a name for your TrajTokenizer!")

        # Name of the tokenizer
        self._name = tokenizer_name
        # Path to the directory containing the TrajTokenizer's data
        self._tokenizer_dir = os.path.join(tokenizers_dir, tokenizer_name)
        # Path to the Hugging Face tokenizer
        self._hf_tokenizer_path = os.path.join(self._tokenizer_dir, hf_tokenizer_dirname)
        # Path to the JSON file containing additional information about the TrajTokenizer
        self._extra_info_path = os.path.join(self._tokenizer_dir, extra_info_filename)

        # Final Hugging Face tokenizer
        self._hf_tokenizer: PreTrainedTokenizerBase = None
        # Boolean indicating whether the TrajTokenizer has been trained or not
        self._is_trained = False

        # HuggingFaceTrajDataset used to train the tokenizer
        self._dataset = dataset
        # Hugging Face dataset associated with this HuggingFaceTrajDataset
        self._hf_dataset = None if dataset is None else dataset.get_hf_dataset()

        # Size of the token vocabulary
        self._vocab_size: int = None
        # Maximum number of tokens per sequence
        self._sequences_max_length: int = None


    @staticmethod
    def get_saved_tokenizers():
        """Returns a dictionary including the paths to the directories of all saved TrajTokenizer.

        The keys are the name of the TrajTokenizer, and the values are the absolute paths to their storage directories.
        """

        # List of the names of all saved TrajTokenizer
        tokenizer_names_list = [tokenizer_name for tokenizer_name in os.listdir(tokenizers_dir)
                                if os.path.isdir(os.path.join(tokenizers_dir, tokenizer_name))]
        # Dictionary that will contain the TrajTokenizer names and the paths to their storage directories
        directories_dict = {tokenizer_name: os.path.join(tokenizers_dir, tokenizer_name)
                            for tokenizer_name in tokenizer_names_list}
        return directories_dict


    @staticmethod
    def load_from_file(tokenizer_dir: str):
        """Loads and returns a TrajTokenizer stored in your local data.

        The local storage directory must contain the following:
            - A Hugging Face tokenizer folder "hf_dataset"
            - An additional information file

        If the dataset previously associated with the saved TrajTokenizer
        has not been saved or has been deleted from your local data, then no
        HuggingFaceTrajDataset will be associated with TrajTokenizer returned by this method.

        Args:
            tokenizer_dir (str): Path to the directory containing the TrajTokenizer.

        Returns:
            TrajTokenizer: Instance of TrajTokenizer loaded from local data.
        """

        if not(os.path.exists(tokenizer_dir)):
            raise FileNotFoundError("Error: No TrajTokenizer is stored here!")
    
        # Load the file that contains additional information
        extra_info_path = os.path.join(tokenizer_dir, extra_info_filename)
        with open(extra_info_path, "r") as f:
            extra_info = json.load(f)

        # Load the HuggingFaceTrajDataset previously associated with the saved Tokenizer
        # if it exists in the local data
        dataset_dir = extra_info["dataset_dir"]
        if os.path.exists(dataset_dir):
            dataset = HuggingFaceTrajDataset.load_from_file(dataset_dir=dataset_dir)
        else:
            dataset = None

        # Retrieve the class of the TrajTokenizer to load
        tokenizer_classes = get_module_classes(sys.modules[__name__])
        tokenizer_class: type[TrajTokenizer] = tokenizer_classes[extra_info["class"]]

        # Load the Hugging Face Tokenizer associated with the TrajTokenizer
        hf_tokenizer_path = os.path.join(tokenizer_dir, hf_tokenizer_dirname)
        hf_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=hf_tokenizer_path)

        # Instantiate the new TrajTokenizer using the loaded information
        tokenizer = tokenizer_class(tokenizer_name=extra_info["name"], dataset=dataset)
        tokenizer._hf_tokenizer = hf_tokenizer
        # Retrieve the token vocabulary size and the maximum token sequence length
        tokenizer._vocab_size = hf_tokenizer.vocab_size
        tokenizer._sequences_max_length = hf_tokenizer.model_max_length
        # Set the "is_trained" attribute to True because every saved TrajTokenizer
        # is associated with a trained Hugging Face tokenizer
        tokenizer._is_trained = True
        
        return tokenizer


    def save(self):
        """Save the TrajTokenizer to a directory that contains all the data related to it
        (e.g., Hugging Face tokenizer, additional information, etc.).
        """

        # Exception if the Hugging Face tokenizer has not been trained
        if not(self._is_trained):
            raise ValueError("Error: You must train your TrajTokenizer (using the 'train' method) "
                             "before saving it to your local data!")

        # Exception if a TrajTokenizer with the same name already exists in the local files
        if os.path.exists(self._tokenizer_dir):
            raise FileExistsError(f"Error: this TrajTokenizer already exists: {self._tokenizer_dir}!\n"
                                  "Delete it from your files or load it using TrajTokenizer.load_from_file method.")
        # Save the Hugging Face tokenizer associated with this TrajTokenizer
        self._hf_tokenizer.save_pretrained(self._hf_tokenizer_path)

        # Save the additional information associated with this TrajTokenizer
        extra_info = {
            "name" : self._name,        # Name of the TrajTokenizer
            "class" : self.__class__.__name__,        # Name of the class of the TrajTokenizer instance
            "dataset_dir" : self._dataset.get_dir(),        # Directory that should contain the HuggingFaceTrajDataset associated with this TrajTokenizer
        }
        with open(self._extra_info_path, "w") as f:
            json.dump(obj = extra_info, fp = f)


    @abstractmethod
    def initialize(self, vocab_size: int, sequences_max_length: int):
        """Creates the initial Hugging Face tokenizer before its training
        by instantiating its initial architecture and specifications.

        You cannot initialize a tokenizer that has been already initialized
        (if the Hugging Face tokenizer has already been instantied).

        Args:
            vocab_size (int, optional): Size of the tokenizer vocabulary.
            sequences_max_length (int, optional): Maximum number of tokens per sequence.
        """

        if self._hf_tokenizer is not None:
            raise ValueError("Error: This TrajTokenizer has already been initialized!")

        # Size of the token vocabulary
        self._vocab_size = vocab_size
        # Maximum number of tokens per sequence
        self._sequences_max_length = sequences_max_length
        

    @abstractmethod
    def train(self):
        """Trains and finalizes the Hugging Face tokenizer associated with this TrajTokenizer.

        You cannot train a tokenizer if:
            - The Hugging Face tokenizer has not been initialized
            - It has already been trained
            - No training HuggingFaceTrajDataset is associated with this TrajTokenizer
        """

        if self._hf_tokenizer is None:
            raise ValueError("Error: You must initialize your TrajTokenizer (using the 'initialize' method) "
                             "before training it!")
        if self._is_trained:
            raise ValueError("Error: The Hugging Face tokenizer associated with this TrajTokenizer has already been trained!")
        if self._dataset is None:
            raise ValueError("Error: You must associate a HuggingFaceTrajDataset with this TrajTokenizer before training it!\n"
                             "Use the set_dataset method.")


    def get_name(self):
        """Returns the name of the TrajTokenizer.
        """

        return self._name
    

    def get_dir(self):
        """Returns the path to the directory that contains the TrajTokenizer data.
        """

        return self._tokenizer_dir


    def get_hf_tokenizer(self):
        """Returns the Hugging Face tokenizer associated with this TrajTokenizer.
        """

        if self._hf_tokenizer is None:
            raise ValueError("Error: This TrajTokenizer has not been initialized!\n"
                             "Use the 'initialize' method.")
        return self._hf_tokenizer
    

    def get_max_position_embeddings(self):
        """Returns the size of the embedding matrix, i.e., the maximum sequence length
        plus the number of special tokens added by the tokenizer."""

        if not(self._is_trained):
            raise ValueError("Error: This TrajTokenizer has not been trained!\n"
                             "Use the 'train' method!")

        return self._sequences_max_length + self._hf_tokenizer.num_special_tokens_to_add()
    

    def get_vocab_size(self):
        """Returns the size of the token vocabulary.
        """

        if self._vocab_size is None:
            raise ValueError("Error: Your TrajTokenizer is not initialized!\n"
                             "Use the 'initialize' method.")

        return self._vocab_size
    

    def set_dataset(self, dataset: HuggingFaceTrajDataset):
        "Associates a training HuggingFaceTrajDataset to the TrajTokenizer."

        self._dataset = dataset


    def tokenize_sequence(self, sequence: Any, truncation=True, padding_method="max_length"):
        """Returns a tokenized version of a given multivariate sequence (with several columns).

        The input text sequence should be contained in a "text" column of the sequence.

        Args:
            sequence (Any): Sequence that contains the text to tokenize (e.g., encoded spatio-temporal sequence).
            truncation (bool, optional): Whether to truncate the sequence if it is too long. Defaults to True.
            padding_method (str, optional): Padding strategy to use. Defaults to "max_length".
        """

        if not(self._is_trained):
            raise ValueError("Error: This TrajTokenizer has not been trained!\n"
                             "Use the 'train' method!")

        return self._hf_tokenizer(text=sequence["text"], truncation=truncation, padding=padding_method,
                               max_length=self._sequences_max_length)
    

    def tokenize_dataset(self, dataset: Dataset, truncation=True, padding_method="max_length"):
        """Returns a tokenized version of a given Hugging Face dataset containing multivariate sequences.

        These sequences must include a "text" column that contains the data to tokenize.

        This method uses the "map" function of the Hugging Face datasets, allowing the application
        of a given function by batch to a dataset.

        Args:
            dataset (Dataset): Dataset that includes the sequences to tokenize.
            truncation (bool, optional): Whether to truncate the sequences that are too long. Defaults to True.
            padding_method (str, optional): Padding strategy to use. Defaults to "max_length".
        """

        if not(self._is_trained):
            raise ValueError("Error: This TrajTokenizer has nos been trained!\n"
                             "Use the 'train' method!")

        # Create the function that will be input to the "map" function to tokenize the dataset by batch
        tokenizer_sequence_function = partial(
            self.tokenize_sequence,
            truncation=truncation,
            padding_method=padding_method,
            )
        
        return dataset.map(tokenizer_sequence_function, batched=True)



class RobertaTrajTokenizer(TrajTokenizer):
    """Class that inherits from TrajTokenizer and represents the Tokenizer used by a Roberta model.

    See the TrajTokenizer documentation for additional information.

    Args:
        tokenizer_name (str, optional): Name of the tokenizer used for storage. Defaults to "roberta_tokenizer".
        dataset (HuggingFaceTrajDataset, optional): Trajectory dataset used to train the tokenizer. Defaults to None.
    """

    def __init__(self, tokenizer_name="roberta_tokenizer", dataset=None):
        super().__init__(tokenizer_name, dataset)


    def initialize(self, vocab_size=52000, sequences_max_length=512):
        super().initialize(vocab_size=vocab_size, sequences_max_length=sequences_max_length)
        # Instantiate the initial Byte-Level Byte Pair Encoding tokenizer
        self._hf_tokenizer = ByteLevelBPETokenizer()

    
    def train(self):
        """Trains a tokenizer based on the Byte-Level Byte Pair Encoding algorithm for the Roberta model.

        This tokenizer uses the following special tokens:
            - unk_token: "unk"
            - pad_token: "pad"
            - cls_token: "s"
            - sep_token: "/s"
            - mask_token: "mask"
        """

        super().train()

        init_tokenizer = self._hf_tokenizer

        # Train the initial tokenizer with the dataset
        init_tokenizer.train_from_iterator(
            batch_iterator(batch_size=1000, dataset=self._hf_dataset),
            vocab_size=self._vocab_size,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )

        # Create the final tokenizer with the correct architecture
        tokenizer = RobertaTokenizerFast(
            tokenizer_object=init_tokenizer._tokenizer,
            # Special tokens
            unk_token="<unk>",
            pad_token="<pad>",
            cls_token="<s>",
            sep_token="</s>",
            mask_token="<mask>"
        )

        # Define the tokenizer's characteristics
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        tokenizer.model_max_length = self._sequences_max_length

        # Define the post-processing method
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", init_tokenizer.token_to_id("</s>")),
            ("<s>", init_tokenizer.token_to_id("<s>")),
        )

        self._hf_tokenizer = tokenizer
        self._is_trained = True



class DebertaTrajTokenizer(TrajTokenizer) :
    """Class that inherits from TrajTokenizer and represents the Tokenizer used by a Deberta model.

    See the TrajTokenizer documentation for additional information.

    Args:
        tokenizer_name (str, optional): Name of the tokenizer used for storage. Defaults to "roberta_tokenizer".
        dataset (HuggingFaceTrajDataset, optional): Trajectory dataset used to train the tokenizer. Defaults to None.
        vocab_size (int, optional): Size of the tokenizer vocabulary. Defaults to 52000.
        sequences_max_length (Maximum number of tokens per sequence, optional): _description_. Defaults to 512.
    """

    def __init__(self, tokenizer_name = "deberta_tokenizer", dataset = None):
        super().__init__(tokenizer_name, dataset)


    def initialize(self, vocab_size=52000, sequences_max_length=512):
        super().initialize(vocab_size=vocab_size, sequences_max_length=sequences_max_length)
        # Instantiate the initial Byte-Level Byte Pair Encoding tokenizer
        self._hf_tokenizer = ByteLevelBPETokenizer()

    
    def train(self):
        """Trains a tokenizer based on the Byte-Level Byte Pair Encoding algorithm for the Deberta model.

        This tokenizer uses the following special tokens:
            - bos_token: "CLS"
            - eos_token: "SEP"
            - sep_token: "SEP"
            - cls_token: "CLS"
            - unk_token: "UNK"
            - pad_token: "PAD"
            - mask_token: "MASK"

        See the documentation of the TrajTokenizer.train method for additional information.
        """

        super().train()

        init_tokenizer = self._hf_tokenizer

        # Train the initial tokenizer with the dataset
        init_tokenizer.train_from_iterator(
            batch_iterator(batch_size=1000, dataset=self._hf_dataset),
            vocab_size=self._vocab_size,
            special_tokens=["[CLS]", "[SEP]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"]
        )
        # Create the final tokenizer with the correct architecture
        tokenizer = DebertaTokenizerFast(
            tokenizer_object=init_tokenizer._tokenizer,
            # Special tokens
            bos_token="[CLS]",
            eos_token="[SEP]",
            sep_token="[SEP]",
            cls_token="[CLS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            mask_token="[MASK]"
        )

        # Define the tokenizer's characteristics
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        tokenizer.model_max_length = self._sequences_max_length

        # Define the post-processing method
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[SEP]", init_tokenizer.token_to_id("[SEP]")),
            ("[CLS]", init_tokenizer.token_to_id("[CLS]")),
        )

        self._hf_tokenizer = tokenizer
        self._is_trained = True