"""
This Python module contains the implementation of the custom Hugging Face datasets used to train our Deep Learning models.

These custom datasets are based on trajectory data from open-source datasets such as Geolife, Gowalla, etc.  
You must download these datasets locally beforehand (see the README for mroe information).
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import os
import json
from abc import ABC, abstractmethod
import pandas as pd

from datasets import Dataset as hugging_face_dataset

# Internal libraries

import sys
sys.path.append("mobiBERT")

from data.datasets.dataloaders import GeolifeDataLoader, GowallaDataLoader
from data.datasets.dataset import Dataset

from pipeline.utils import get_module_classes
from pipeline.hf_traj_datasets.utils import datasets_dir, hf_dataset_dirname, extra_info_filename
from pipeline.hf_traj_datasets.trajectory_processors import round_timestamp, geohash_encode, create_sequences

# --------------------------------------------------------- CLASS IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

class HuggingFaceTrajDataset(ABC):
    """Class that implements a custom Hugging Face dataset associated with a trajectory dataset (Geolife, Gowalla, etc.).  

    Each instance of this class allows retrieving the Pandas dataframe corresponding to the initial trajectory data,
    the dataframe of processed data (e.g., spatio-temporal data converted into spatio-temporal sequences composed of encoded coordinates)
    and the Hugging Face dataset version of the processed data.  

    Each HuggingFaceTrajDataset can be saved to a directory within your local files, with the
    same name as "dataset_name".  
    However, if your try to save a HuggingFaceTrajDataset for which the final Hugging Face dataset
    (the "hf_dataset" attribute) has not been created, it will be automatically instantiated by the "save" method.

    This Hugging Face dataset aims to be used by a Deep Learning model (encoder).  

    The trajectory data must include the following columns:  
        - Timestamp: "ts"  
        - User ID: "uid"  
        - Latitude: "latitude"  
        - Longitude: "longitude"  

    Args:
        dataset_name (str): Name of the Hugging Face dataset used for storage.
    """

    def __init__(self, dataset_name: str):
        if dataset_name is None:
            raise ValueError("Error: You must specify a name for your HuggingFaceTrajDataset!")
        # Name of the dataset
        self._name = dataset_name
        # Directory containing the data associated with the HuggingFaceTrajDataset
        self._dataset_dir = os.path.join(datasets_dir, dataset_name)
        # Path to the Hugging Face dataset
        self._hf_dataset_path = os.path.join(self._dataset_dir, hf_dataset_dirname)
        # Path to the JSON file containing additional information about the HuggingFaceTrajDataset
        self._extra_info_path = os.path.join(self._dataset_dir, extra_info_filename)
        
        # Initial trajectory dataframe before processing it to create its Hugging Face version
        self._initial_dataframe: pd.DataFrame = None
        # Processed trajectory dataframe ready to be transform to a Hugging Face dataset
        self._processed_dataframe: pd.DataFrame = None
        # Hugging Face dataset that contains sequences that will be used as input of DL models
        self._hf_dataset: Dataset = None
        # Boolean indicating whether the Hugging Face associate with
        # this HuggingFaceTrajDataset has been created or not
        self._is_hf_dataset_created = False

        # Dictionnary that maps each user ID to an index (ranging from 0 to the number of unique users)
        self._labels_to_ids: dict[str, int] = None
        # Same dictionnary but with keys and values reversed
        self._ids_to_labels: dict[int, str] = None


    @staticmethod
    def get_saved_datasets():
        """Returns a dictionary including the paths to the directories of all saved HuggingFaceTrajDataset.

        The keys are the name of the HuggingFaceTrajDatasets, and the values are the absolute paths to their storage directories.
        """

        # List of the names of all saved HuggingFaceTrajDataset
        dataset_names_list = [dataset_name for dataset_name in os.listdir(datasets_dir)
                             if os.path.isdir(os.path.join(datasets_dir, dataset_name))]
        # Dictionary that will contain the HuggingFaceTrajDataset names and the paths to their storage directories
        directories_dict = {dataset_name: os.path.join(datasets_dir, dataset_name)
                          for dataset_name in dataset_names_list}
        return directories_dict
        


    @staticmethod
    def load_from_file(dataset_dir: str):
        """Loads and returns a HuggingFaceTrajDataset stored in your local data.

        The local storage directory must contain the following:
            - A Hugging Face dataset folder "hf_dataset"
            - An additional information file

        Args:
            dataset_dir (str): Path to the directory containing the HuggingFaceTrajDataset.

        Returns:
            HuggingFaceTrajDataset: Instance of HuggingFaceTrajDataset loaded from local data.
        """

        if not(os.path.exists(dataset_dir)):
            raise FileNotFoundError("Error: No HuggingFaceTrajDataset is stored here!")

        # Load the file that contains additional information
        extra_info_path = os.path.join(dataset_dir, extra_info_filename)
        with open(extra_info_path, "r") as f:
            extra_info = json.load(f)
        
        # Retrieve the class of the HuggingFaceTrajDataset to load
        dataset_classes = get_module_classes(sys.modules[__name__])
        dataset_class: type[HuggingFaceTrajDataset] = dataset_classes[extra_info["class"]]
        # Instantiate the HuggingFaceTrajDataset using its associated class and name
        dataset = dataset_class(dataset_name=extra_info["name"])
        
        # Load the Hugging Face dataset associated with this HuggingFaceTrajDataset
        hf_dataset_path = os.path.join(dataset_dir, hf_dataset_dirname)
        dataset._hf_dataset = hugging_face_dataset.load_from_disk(dataset_path=hf_dataset_path)
        # Set the "is_hf_dataset_created" attribute to True because every saved
        # HuggingFaceTrajDataset is associated with an existing Hugging Face dataset
        dataset._is_hf_dataset_created = True

        return dataset
    

    def save(self):
        """Saves the HuggingFaceTrajDataset to a directory that contains all the data related to the used dataset.

        This method also saves additional information about the dataset, such as its specified name, the name of its class, etc.  
        You can't save your HuggingFaceTrajDataset another one with the same name already exists.
        """

        # Create the Hugging Face dataset if it doesn't exist
        if not(self._is_hf_dataset_created):
            self._create_hf_dataset()

        # Save the HuggingFaceTrajDataset: Exception if it already exists
        if os.path.exists(self._dataset_dir):
            raise FileExistsError(f"Error: this HuggingFaceTrajDataset already exists: {self._dataset_dir}!\n"
                                  "Delete it from your files or load it using HuggingFaceTrajDataset.load_from_file method.")
        # Save the Hugging Face dataset
        self._hf_dataset.save_to_disk(dataset_path=self._hf_dataset_path)

        # Save the additional information in a JSON file
        extra_info = {
            "name" : self._name,        # Name of the HuggingFaceTrajDataset
            "class" : self.__class__.__name__       # Name of the class of the HuggingFaceTrajDataset
        }
        with open(self._extra_info_path, "w") as f:
            json.dump(obj = extra_info, fp = f)
    

    @abstractmethod
    def _load_data(self):
        """Loads a dataframe containing organized trajectory data from a trajectory dataset.

        The data are loaded using the "data" package of this library.  
        This package handles the loading and preprocessing of raw trajectory datasets.

        The trajectory data must include the following columns:
            - Timestamp: "ts"
            - User ID: "uid"
            - Latitude: "latitude"
            - Longitude: "longitude"
        """

        print("\nLOADING THE INITIAL DATA...\n")
    

    @abstractmethod
    def _process_data(self):
        """Processes the initial trajectory data (e.g., simplifies trajectories, creates sequences (time series) usable by the model, etc.).

        The processed dataframe is ready to be transformed into its Hugging Face version.
        """

        # Load the initial data if it doesn't exist
        if self._initial_dataframe is None :
            self._load_data()

        print("\nPROCESSING THE INITIAL DATA...\n")
    

    def _create_hf_dataset(self):
        """Transforms the Pandas dataframe containing processed trajectories (time series) into its Hugging Face dataset version.
        """

        # Exception if the Hugging Face dataset has already been created
        if self._is_hf_dataset_created:
            raise ValueError("Error: The Hugging Face dataset associated with this HuggingFaceTrajDataset has already been created!")
        # Process the initial data if necessary
        if self._processed_dataframe is None:
            self._process_data()

        print("\nCREATING THE HUGGING FACE DATASET...\n")

        self._hf_dataset = hugging_face_dataset.from_pandas(self._processed_dataframe)
        self._is_hf_dataset_created = True


    def get_name(self):
        """Returns the name of the HuggingFaceTrajDataset.
        """

        return self._name
    

    def get_initial_dataframe(self):
        """Returns the Pandas dataframe containing the initial trajectory data (before processing).
        """

        # Load the initial data if it hasn't already been done
        if self._initial_dataframe is None:
            self._load_data()

        return self._initial_dataframe
    

    def get_processed_dataframe(self):
        """Returns the Pandas dataframe containing the processed data.
        
        This dataframe is ready to be transform into its Hugging Face dataset version.
        """

        # Process the initial data if necessary
        if self._processed_dataframe is None:
            self._process_data()
        
        return self._processed_dataframe
    

    def get_hf_dataset(self):
        """Returns the Hugging Face dataset containing processed trajectory data.
        """

        # Create the Hugging Face dataset if it doesn't exist
        if self._hf_dataset is None:
            self._create_hf_dataset()
        
        return self._hf_dataset
    

    def get_users(self):
        """Returns a list containing all unique user IDs included in the created Hugging Face dataset.
        """

        # Create the Hugging Face dataset if it doesn't exist
        if self._hf_dataset is None:
            self._create_hf_dataset()

        return list(set(self._hf_dataset["label"]))
    

    def get_nb_users(self):
        """Returns the number of unique user associated with the spatio-temporal sequences (trajectories) in the created Hugging Face dataset.
        """

        return len(self.get_users())
    

    def get_labels_to_ids(self):
        """Returns a dictionnary that maps each user ID to an index (ranging from 0 to the number of unique users).
        """

        # Create the Hugging Face dataset if it doesn't exist
        if self._hf_dataset is None:
            self._create_hf_dataset()
            
        # Create the dictionnary if necessary
        if self._labels_to_ids is None:
            # The keys of the dictionnary are the user IDs and the values are the indexes (ranging from 0 to the number of unique users)
            self._labels_to_ids = {uid: index for index, uid in enumerate(self.get_users())}
        
        return self._labels_to_ids
    

    def get_ids_to_labels(self):
        """Returns a dictionnary that maps each index (ranging from 0 to the number of unique users) to a user ID.
        """

        # Create the Hugging Face dataset if it doesn't exist
        if self._hf_dataset is None:
            self._create_hf_dataset()
        
        # Create the dictionnary if necessary
        if self._ids_to_labels is None:
            # The keys of the dictionnary are the indexes (ranging from 0 to the number of unique users) and the values are the user IDs
            self._ids_to_labels = {index: uid for uid, index in self.get_labels_to_ids().items()}
        
        return self._ids_to_labels
        

    def get_dir(self):
        """Returns the path to the directory that contains the HuggingFaceTrajDataset.
        """
        return self._dataset_dir



class GeolifeGeoHashed(HuggingFaceTrajDataset):
    """Class that inherits from HuggingFaceTrajDataset and implements a custom Hugging Face dataset
    associated with the Geolife open-source dataset.

    Its main specificity lies in the data processing: trajectories are
    encoded using the GeoHash method.

    See the HuggingFaceTrajDataset documentation for additional information.

    Args:
        dataset_name (str, optional): Name of the Hugging Face dataset used for storage. Defaults to "geolife_geohashed".
    """

    def __init__(self, dataset_name: str = "geolife_geohashed"):
        super().__init__(dataset_name = dataset_name)

    
    def _load_data(self):
        super()._load_data()

        # Load the raw Geolife dataset using the Dataset class from "data" package
        raw_dataset = Dataset(name='geolife', dataloader=GeolifeDataLoader())
        trajectory_dataframe = raw_dataset.get_dataframe()
        self._initial_dataframe = trajectory_dataframe
    

    def _process_data(self):
        super()._process_data()

        # Simplify trajectories by "merging" those that are temporally close
        processed_trajectories = round_timestamp(dataframe=self._initial_dataframe, timestamp_rounding=120)

        # Encode the longitudes and latitudes using the GeoHash method
        processed_trajectories = geohash_encode(dataframe=processed_trajectories, encoding_precision=8)

        # Create spatio-temporal time series (trajectories) using the "sliding window" technique
        processed_trajectories = create_sequences(processed_trajectories, temporal_window=3*60*60, temporal_lag=20*60, use_threading=True)

        self._processed_dataframe = processed_trajectories



class GowallaGeoHashed(HuggingFaceTrajDataset):
    """Class that that inherits from HuggingFaceTrajDataset and implements a custom Hugging Face dataset
    associated with the Gowalla open-source dataset.

    This class inherits from HuggingFaceTrajDataset.  
    Its main specificity lies in the data processing: trajectories are
    encoded using the GeoHash method.

    See the HuggingFaceTrajDataset documentation for additional information.

    Args:
        dataset_name (str, optional): Name of the Hugging Face dataset used for storage. Defaults to "gowalla_geohashed".
    """

    def __init__(self, dataset_name: str = "gowalla_geohashed"):
        super().__init__(dataset_name = dataset_name)


    def _load_data(self):
        super()._load_data()

        # Load the raw Gowalla dataset using the Dataset class from "data" package
        dataloader = Dataset(name="gowalla", dataloader=GowallaDataLoader())
        trajectory_dataframe = dataloader.get_dataframe()
        self._initial_dataframe = trajectory_dataframe
    
    
    def _process_data(self):
        super()._process_data()

        # Simplify trajectories by "merging" those that are temporally close
        processed_trajectories = round_timestamp(dataframe=self._initial_dataframe, timestamp_rounding=120)

        # Encode the longitudes and latitudes using the GeoHash method
        processed_trajectories = geohash_encode(dataframe=processed_trajectories, encoding_precision=8)

        # Create spatio-temporal time series (trajectories) using the "sliding window" technique
        processed_trajectories = create_sequences(processed_trajectories, temporal_window=3*60*60, temporal_lag=20*60, use_threading=True)

        self._processed_dataframe = processed_trajectories