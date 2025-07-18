from abc import ABC, abstractmethod
from ..datasets.dataset import Dataset
from typing import List
import os
import pandas as pd

class IFilter(ABC):
    @abstractmethod
    def filter(self, dataframe): raise NotImplementedError("filter method must be implemented!")

class Filterer:
    def __init__(self, name:str, dataset:Dataset=None, filters:List[IFilter]=[]):
        self.name = name
        self.__dataset = dataset.clone()
        self.filters = filters
        self.__data_dir = dataset.get_data_dir()
        self.__filtred_dataset_filename = os.path.join(self.__data_dir, self.name + '.parquet')
        self.__filtered_dataframe = dataset.get_dataframe()

    def filter(self):
        for filter in self.filters:
            self.__filtered_dataframe = filter.filter(self.__filtered_dataframe)
        self.__filtered_dataframe.to_parquet(self.__filtred_dataset_filename)
        self.__dataset.set_trajectories_df_filename(self.__filtred_dataset_filename)
        return self.__dataset

    
class DummyFilter(IFilter):
        def filter(self, dataframe): return dataframe

class Year2009Filter(IFilter):
    def filter(self, dataframe):
        return dataframe[(dataframe['ts'] < 1262304000)  & (dataframe['ts'] > 1230768000)].reset_index(drop=True, inplace=False)
    
class TemporalIntervalFilter(IFilter):
    def __init__(self, start_ts, end_ts):
        self.start_ts = start_ts
        self.end_ts = end_ts

    def filter(self, dataframe):
        return dataframe[(dataframe['ts'] < self.end_ts)  & (dataframe['ts'] > self.start_ts)].reset_index(drop=True, inplace=False)
    
class MinNbLocationsFilter(IFilter):
    def __init__(self, min_nb_locations):
        self.min_nb_locations = min_nb_locations

    def filter(self, dataframe):
        nb_location_by_tid = dataframe.groupby('tid').size()
        tids = nb_location_by_tid[nb_location_by_tid >= self.min_nb_locations].index
        return dataframe[dataframe['tid'].isin(tids)].reset_index(drop=True, inplace=False)
    
class MinDurationFilter(IFilter):
    def __init__(self, min_duration):
        self.min_duration = min_duration

    def filter(self, dataframe):
        duration_by_tid = dataframe.groupby('tid').apply(lambda x: x['ts'].max() - x['ts'].min())
        tids = duration_by_tid[duration_by_tid >= self.min_duration].index
        return dataframe[dataframe['tid'].isin(tids)].reset_index(drop=True, inplace=False)
    
class UserIdsFilter(IFilter):
    def __init__(self, user_ids):
        self.user_ids = user_ids

    def filter(self, dataframe):
        return dataframe[dataframe['uid'].isin(self.user_ids)].reset_index(drop=True, inplace=False)
    
class WeekdayFilter(IFilter):
    def __init__(self, weekday_name):
        weekday_names = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY','SUNDAY']
        weekday_name = weekday_name.upper()

        if weekday_name not in weekday_names:
            raise Exception(f"weekday_name must be one of {weekday_names}")
        self.weekday = weekday_names.index(weekday_name)


    def filter(self, dataframe):
        tid_by_weekday = dataframe.groupby('tid').apply(lambda x: pd.to_datetime(x['ts'], unit='s').dt.weekday.mode()[0])
        tids = tid_by_weekday[tid_by_weekday == self.weekday].index.values
        return dataframe[dataframe['tid'].isin(tids)].reset_index(drop=True, inplace=False)
    
class WeekdaysFilter(IFilter):
    def __init__(self, filter_weekday_names):
        filter_weekday_names = [weekday_name.upper() for weekday_name in filter_weekday_names]
        filter_weekday_names = list(set(filter_weekday_names))
        n = len(filter_weekday_names)
        weekday_names = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY','SUNDAY']
        filter_weekday_names = [filter_weekday_name for filter_weekday_name in filter_weekday_names if filter_weekday_name in weekday_names]
        m = len(filter_weekday_names)
        if n != m:
            print(f"Warning: {n-m} weekday_names are not in {weekday_names}")
        self.weekdays = [weekday_names.index(weekday_name) for weekday_name in filter_weekday_names]

    def filter(self, dataframe):
        tid_by_weekday = dataframe.groupby('tid').apply(lambda x: pd.to_datetime(x['ts'], unit='s').dt.weekday.mode()[0])
        tids = tid_by_weekday[tid_by_weekday.isin(self.weekdays)].index.values
        return dataframe[dataframe['tid'].isin(tids)].reset_index(drop=True, inplace=False)
    


