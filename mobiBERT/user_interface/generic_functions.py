"""
This module contains all the generic functions of the User Interface program
"""

#----------------------------------------------------- Importing libraries -------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("mobiBERT")

import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly
from typing import Literal

# Adding the packages of our library
from data.datasets.dataloaders import GeolifeDataLoader, TDriveDataLoader, GowallaDataLoader
from data.datasets.dataset import Dataset
from data.preprocessors.filterers import TemporalIntervalFilter, WeekdaysFilter

#----------------------------------------------------- Definition of variables -------------------------------------------------------------------------------------------------------------------

# Timestamp of "2000-01-01 00:00:00"
min_date_2000 = "2000-01-01 00:00:00"

# Timestamp of today's date
today_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#--------------------------------------------------- Implementation of functions -------------------------------------------------------------------------------------------------------------------

def import_tdrive() -> pd.DataFrame :
    tdrive_dataset = Dataset(name='tdrive', dataloader = TDriveDataLoader())
    tdrive_df = tdrive_dataset.get_dataframe()

    return tdrive_df


def import_geolife() -> pd.DataFrame :
    geolife_dataset = Dataset(name='geolife', dataloader = GeolifeDataLoader())
    geolife_df = geolife_dataset.get_dataframe()
    
    return geolife_df

def import_gowalla() -> pd.DataFrame:
    gowalla_dataset = Dataset(name="gowalla", dataloader=GowallaDataLoader())
    gowalla_df = gowalla_dataset.get_dataframe()

    return gowalla_df


def timestamp_to_str(timestamp : float) -> str :
    to_datetime = datetime.fromtimestamp(timestamp)
    to_str = str(to_datetime)
    return to_str


def str_to_timestamp(date_str : str) -> float :
    if date_str is None :
        return date_str
    
    to_datetime = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    to_timestamp = to_datetime.timestamp()
    return to_timestamp


def update_dates(dataframe : pd.DataFrame, current_date : None | float) :
    min_possible_date = dataframe.ts.min() if len(dataframe) > 0 else str_to_timestamp(min_date_2000)
    max_possible_date = dataframe.ts.max() if len(dataframe) > 0 else str_to_timestamp(today_date)

    new_date = min_possible_date if current_date is None else current_date

    new_date = max(min_possible_date, min(new_date, max_possible_date))

    return min_possible_date, max_possible_date, new_date


def update_min_nb_locations(dataframe : pd.DataFrame, current_min_nb_locations : int) :
    # Calculation of the minimum and maximum number of locations per trajectory
    nb_location_by_tid = dataframe.groupby('tid').size()
    min_nb_locations_possible  = nb_location_by_tid.min()
    max_nb_locations_possible  = nb_location_by_tid.max()

    # Adjustment of the minimum number of locations selected by the user according to its value and
    # the minimum and maximum number of locations possibles
    min_nb_locations = min(max(int(current_min_nb_locations), min_nb_locations_possible), max_nb_locations_possible)

    return min_nb_locations_possible, max_nb_locations_possible, min_nb_locations


def update_min_duration(dataframe : pd.DataFrame, current_min_duration : int) :
    if len(dataframe) > 0 :
        duration_by_tid = dataframe.groupby('tid').apply(lambda x: x['ts'].max() - x['ts'].min())
    else :
        duration_by_tid = pd.Series(0)
    min_duration_possible = int(duration_by_tid.min()/60)
    max_duration_possible = int(duration_by_tid.max()/60)

    min_duration = min(max(float(current_min_duration), min_duration_possible), max_duration_possible)

    return min_duration_possible, max_duration_possible, min_duration


def get_time_filter(date_type : str, weekdays : list[str], precise_date : float | None, time_range : float) :
        if date_type == "Select weekday(s)" :
            time_filter = WeekdaysFilter(weekdays)

        elif date_type == "Select a precise date" :
            time_range_second = time_range*86400 # Conversion of days to seconds
            time_filter = TemporalIntervalFilter(precise_date, precise_date + time_range_second)

        return time_filter


def create_map(dataframe : pd.DataFrame, dataset_type : Literal["T-Drive", "Geolife", "Gowalla"]) :
    # Map that contains the trajectories
    fig = px.line_map(dataframe, lat = "latitude", lon = "longitude", color = "uid", line_group = "tid", height = 800,
                      map_style = "open-street-map", title = f"Map of the trajectories from the {dataset_type} dataset")
    fig.update_traces(line = dict(width = 2.5))

    # Updating the final figure
    fig.update_layout(margin = dict(b = 0, r = 0, t = 10, l = 0), title = dict(x = 0.5, font = dict(color = "black")),
                    showlegend = True)
        
    return fig