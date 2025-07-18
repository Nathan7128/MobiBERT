from .dataloaders import DataLoader
import pandas as pd
import os

class Dataset:
    def __init__(self, name:str=None, dataloader:DataLoader=None, trajectories_df_filename:str=None):
        if name is None and dataloader is None and trajectories_df_filename is None:
            raise Exception("Dataset must be initialized with a name, a dataloader or a dataframe filename!")
        
        if trajectories_df_filename is not None:
            self.__trajectories_df_filename = trajectories_df_filename
            self.name = os.path.basename(os.path.dirname(os.path.dirname(trajectories_df_filename)))
            self.__data_dir = os.path.dirname(os.path.dirname(trajectories_df_filename))
            return

        self.name = name
        self.__data_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),\
                                        'data_store', self.name, "trajectories", "data")
        if not os.path.exists(self.__data_dir): os.makedirs(self.__data_dir)
        dataloader.initialize(self.__data_dir)
        self.__trajectories_df_filename = dataloader.load()

    def get_data_dir(self): return self.__data_dir

    def set_trajectories_df_filename(self, filename): 
        self.__trajectories_df_filename = filename

    def get_dataframe(self, lib='pandas'):
        if lib=='pandas': return pd.read_parquet(self.__trajectories_df_filename)

    def clone(self):
        return Dataset(trajectories_df_filename=self.__trajectories_df_filename)