import os, zipfile, tempfile, gzip
from tqdm import tqdm
import pandas as pd
from abc import ABC, abstractmethod


class DataLoader(ABC):
    def __init__(self):
        self.__data_dir = None
        self.__dataframe_file_path = None

    def initialize(self, data_dir:str):
        self.__data_dir = data_dir
        self.__dataframe_file_path = os.path.join(self.__data_dir, "all.parquet")


    def load(self):
        if not os.path.exists(self.__dataframe_file_path): 
            print(f"Dataset does not exist. Downloading...")
            self.download(self.__dataframe_file_path )
        return self.__dataframe_file_path

    @abstractmethod
    def download(self, dataframe_file_path): raise NotImplementedError("download method must be implemented!")



## Geolife Dataset Loader ##
class GeolifeDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.__source_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data_store', 'local_data', "Geolife Trajectories 1.3.zip")

    def download(self, dataframe_file_path):
        z = zipfile.ZipFile(self.__source_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            z.extractall(temp_dir)
            data_dir = os.path.join(temp_dir, "Geolife Trajectories 1.3", "Data")
            trajectory_filenames = [os.path.join(directory, filename) for directory, subdirectories, filenames in os.walk(data_dir) for filename in filenames if filename.endswith('.plt')]
            trajectories = [self.__get_trajectory(filename) for  filename in tqdm(trajectory_filenames)]
            trajectories_df = pd.concat(trajectories)
            trajectories_df.to_parquet(dataframe_file_path)

    
    def __get_trajectory(self, trajectory_filename:str):
        df = pd.read_csv(trajectory_filename, header=None, skiprows=6, usecols=[0, 1, 5, 6])
        df.columns=['latitude', 'longitude', 'date', 'time']
        separator = os.sep
        uid = trajectory_filename.split(separator)[-3]
        tid = f"{trajectory_filename.split(separator)[-3]}-{trajectory_filename.split(separator)[-1].replace('.plt', '')}"

        df['uid'] = uid
        df['tid'] = tid

        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time']) 
        df['ts'] = df['datetime'].astype(int) / 10**9
        df.drop(columns=['date', 'time', 'datetime'], inplace=True)
        df = df[['uid', 'tid', 'ts', 'latitude', 'longitude']]
        df.sort_values(by=['uid', 'tid', 'ts'], inplace=True)
        return df
    

## T-Drive Dataset Loader ##
class TDriveDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.__source_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data_store', 'local_data', "T-drive Taxi Trajectories.zip")

    def download(self, dataframe_file_path):
        z = zipfile.ZipFile(self.__source_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            z.extractall(temp_dir)
            data_dir = os.path.join(temp_dir, "release", "taxi_log_2008_by_id")
            trajectory_filenames = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.txt')]
            
            trajectories = [self.__get_trajectory(filename) for  filename in tqdm(trajectory_filenames)]
            trajectories = [t for t in trajectories if t is not None]

            trajectories_df = pd.concat(trajectories)
            trajectories_df.to_parquet(dataframe_file_path)


    def __get_trajectory(self, trajectory_filename:str):
        try:
            df = pd.read_csv(trajectory_filename)
            df.columns=['uid', 'datetime', 'longitude', 'latitude']
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['tid'] = df['uid'].astype(str) + '-' + df['datetime'].dt.date.astype(str)
            df['ts'] = df['datetime'].astype(int) / 10**9
            df.drop(columns=['datetime'], inplace=True)
            df = df[['uid', 'tid', 'ts', 'latitude', 'longitude']]
            df.sort_values(by=['uid', 'tid', 'ts'], inplace=True)
            return df
        except:
            return None
        
    
## Gowalla Dataset Loader ##
class GowallaDataLoader(DataLoader):
    def __init__(self):
        super().__init__()
        self.__source_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data_store', 'local_data', "loc-gowalla_totalCheckins.txt.gz")

    def download(self, dataframe_file_path):
        with gzip.open(self.__source_path, 'rt', encoding='utf-8') as f:
            trajectories_df = pd.read_csv(f, sep = "\t", header = None)

            trajectories_df.columns = ['uid', 'datetime', 'latitude', 'longitude', 'location_id']

            trajectories_df = self.__process_trajectories(trajectories_df)

            trajectories_df["tid"] = trajectories_df["uid"]

            trajectories_df.to_parquet(dataframe_file_path)

    def __process_trajectories(self, trajectories_df) :
        beijing_lat_min, beijing_lat_max = 39.4, 41.6
        beijing_lon_min, beijing_lon_max = 115.7, 117.4
        trajectories_df = trajectories_df[(trajectories_df['latitude'] >= beijing_lat_min) & (trajectories_df['latitude'] <= beijing_lat_max) &
                                (trajectories_df['longitude'] >= beijing_lon_min) & (trajectories_df['longitude'] <= beijing_lon_max)]    
        
        trajectories_df['datetime'] = pd.to_datetime(trajectories_df['datetime'])
        trajectories_df['ts'] = trajectories_df['datetime'].astype(int) / 10**9
        trajectories_df.drop(columns=['datetime', "location_id"], inplace=True)

        trajectories_df = trajectories_df[['uid', 'ts', 'latitude', 'longitude']]
        trajectories_df.sort_values(by=['uid', 'ts'], inplace=True)
        trajectories_df.reset_index(drop = True, inplace = True)

        return trajectories_df