"""
This module contains all the event listeners functions of the User Interface program
"""

#----------------------------------------------------- Importing libraries -------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append("mobiBERT")

import gradio as gr
from typing import Literal
import pandas as pd

# Adding the packages of our library
from generic_functions import min_date_2000, timestamp_to_str, str_to_timestamp, update_min_duration, update_min_nb_locations
from generic_functions import update_dates, get_time_filter, create_map
from data.preprocessors.filterers import UserIdsFilter, MinNbLocationsFilter, MinDurationFilter

#--------------------------------------------------- Implementation of functions -------------------------------------------------------------------------------------------------------------------

# Reinitialization of components when the dataset is changed
def reset_components(initial_data : pd.DataFrame) :
    return (pd.DataFrame(columns = initial_data.columns), # filtered_by_users_data
            pd.DataFrame(columns = initial_data.columns), # final_data
            gr.update(visible = False, value = None), # date_type
            gr.update(visible = False, value = []), # weekdays
            gr.update(visible = False, value = min_date_2000), # precise_date
            gr.update(visible = False, value = 0.5), # time_range
            gr.update(visible = False, value = 1), # min_nb_locations
            gr.update(visible = False, value = 0)) # min_duration


def switch_dataset(dataset_type : Literal["T-Drive", "Geolife", "Gowalla"], datasets_dictionnary : dict[str, pd.DataFrame]) :
    # Instanciation of the dataframe according to the user's choice
    dataframe = datasets_dictionnary[dataset_type]

    # List that contains all the (unique) user ids from the selected dataset
    unique_user_ids_list = dataframe.uid.unique().tolist()

    return (dataframe, gr.update(choices = unique_user_ids_list, value = [], visible = True))


def update_user_ids(initial_data : pd.DataFrame, user_ids_list : list[int],
                    precise_date : str, min_nb_locations : int, min_duration : int) :
    # Filtering of user ids
    filtered_data = UserIdsFilter(user_ids_list).filter(initial_data)

    # Extraction of the minimum and maximum dates for the selected users
    # We also update the minimum and maximum dates currently selected if they are outside possible dates
    min_possible_date, max_possible_date, precise_date = update_dates(filtered_data, str_to_timestamp(precise_date))

    # Calculation of the minimum and maximum number of locations per trajectory for the selected users
    result_update_min_nb_locations = update_min_nb_locations(filtered_data, min_nb_locations)
    min_nb_locations_possible, max_nb_locations_possible, min_nb_locations = result_update_min_nb_locations

    # Calculation of the minimum and maximum duration for the trajectories for the selected users
    result_update_min_duration = update_min_duration(filtered_data, min_duration)
    min_duration_possible, max_duration_possible, min_duration = result_update_min_duration

    component_visibility = len(user_ids_list) > 0

    return (filtered_data, # filtered_by_users_data
        gr.update(visible = component_visibility), # date_type
        gr.update(value = precise_date, label = f"""Min : {timestamp_to_str(min_possible_date)} |
                                                    Max : {timestamp_to_str(max_possible_date)}"""), # precise_date
        gr.update(visible = component_visibility, minimum = min_nb_locations_possible, maximum = max_nb_locations_possible,
                    value = min_nb_locations), # min_nb_locations
        gr.update(visible = component_visibility, minimum = min_duration_possible, maximum = max_duration_possible,
                    value = min_duration)) # min_duration


def change_date_type(date_type : Literal["Select weekday(s)", "Select a precise date"]) :
    return (gr.update(visible = date_type == "Select weekday(s)"),
            gr.update(visible = date_type == "Select a precise date"),
            gr.update(visible = date_type == "Select a precise date"))


def filter_data(filtered_data : pd.DataFrame, date_type : Literal["Select weekday(s)", "Select a precise date"], 
                weekdays : list[str], precise_date : str, time_range : float, min_nb_locations : int, min_duration : int) :
    if len(filtered_data) == 0 or date_type is None :
        return filtered_data, precise_date
    
    # Modifying precise_date if its value is not valid
    _, _, precise_date = update_dates(filtered_data, str_to_timestamp(precise_date))
    
    # Creating the time filter based on date type
    time_filter = get_time_filter(date_type, weekdays, precise_date, time_range)

    # List of filters to apply to the data
    filters = [time_filter, MinNbLocationsFilter(int(min_nb_locations)),
                MinDurationFilter(int(float(min_duration))*60)]
    # Applying the filters
    for filter in filters :
        filtered_data = filter.filter(filtered_data)

    return filtered_data, precise_date


def display_map(final_data : pd.DataFrame, date_type : Literal["Select weekday(s)", "Select a precise date"],
                dataset_type : Literal["T-Drive", "Geolife", "Gowalla"]) :
    map_visibility = (len(final_data)) > 0 and (not(date_type is None))
    map_figure = create_map(final_data, dataset_type)

    return gr.update(visible = map_visibility, value = map_figure)