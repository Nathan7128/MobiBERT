#----------------------------------------------------- Importing libraries -------------------------------------------------------------------------------------------------------------------

import gradio as gr
import pandas as pd

# Importing functions from the "generic_functions.py" module
from generic_functions import import_tdrive, import_geolife, import_gowalla, min_date_2000

# Importing functions from the "event_listeners.py" module
from event_listeners import switch_dataset, reset_components, update_user_ids, change_date_type, filter_data, display_map

#------------------------------------------------- User Interface implementation -------------------------------------------------------------------------------------------------------------------

with gr.Blocks() as demo :
# 1) Instanciation of State variables

    # Dictionnary that contains the initial dataframes
    datasets_dictionnary = gr.State({"T-Drive" : import_tdrive(), "Geolife" : import_geolife(),
                                     "Gowalla" : import_gowalla()})
    # Dataframe that contains all the initial data from the selected dataset
    initial_data = gr.State(pd.DataFrame())
    # Dataframe that contains the data filtered by the selected user ids
    columns_data = datasets_dictionnary.value["Geolife"].columns
    filtered_by_users_data = gr.State(pd.DataFrame(columns = columns_data))
    # Dataframe obtained after applying all filters to the initial dataframe
    final_data = gr.State(pd.DataFrame(columns = columns_data))


# 2) Creation of Gradium components

    # Creation of the sidebars that contains the filters
    with gr.Sidebar(width = 430) :
        # Set the header of the sidebar
        gr.Markdown(value = "<div style='text-align: center; font-size: 20px;'>Filters</div>")

        # Choice of dataset
        with gr.Row() :
            dataset_type = gr.Radio(choices = datasets_dictionnary.value.keys(), label = "Dataset")

        # Choice of user ids
        with gr.Row() :
            user_ids_list = gr.Dropdown(visible = False, multiselect = True, label = "Users")

        with gr.Column() :
            # Choice of date type for filtering the data
            date_type = gr.Radio(["Select weekday(s)", "Select a precise date"], visible = False)

            # Choice of weekdays if it is the selected date type
            weekdays = gr.Dropdown(visible = False, multiselect = True, label = "Weekdays",
                                   choices = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY','SUNDAY'])
            
            # Choice of a precise date if it is the selected date type
            precise_date = gr.DateTime(type = "string", visible = False, value = min_date_2000, label = "Date")

            time_range = gr.Slider(minimum = 0.5, maximum = 31, label = "Number of days to be selected after this date",
                                   step = 0.5, visible = False)


        # Choice of minimal number of locations (positions) per trajectory
        with gr.Row() :
            min_nb_locations = gr.Slider(label = "Minimum number of locations per trajectory", step = 1,
                                     visible = False, value = 1, show_reset_button = False)
        
        # Choice of the minimum duration of the trajectories (in minutes)
        with gr.Row() :    
            min_duration = gr.Slider(label = "Minimum duration of the trajectories (in minutes)", step = 1,
                                 visible = False, value = 0, show_reset_button = False)

    # Map of the trajectories
    trajectories_map = gr.Plot(visible = False)


# 3) Definition of the event listener functions

    # I) Change of dataset type
    dataset_type.change(fn = switch_dataset, inputs = [dataset_type, datasets_dictionnary],
                        outputs = [initial_data, user_ids_list]).then(fn = reset_components, inputs = initial_data,
                        outputs = [filtered_by_users_data, final_data, date_type, weekdays,
                                   precise_date, time_range, min_nb_locations, min_duration])
    

    # II) Filtering of user ids from the data and deduction of possible values for the other filters
    user_ids_list.change(fn = update_user_ids, inputs = [initial_data, user_ids_list, precise_date, min_nb_locations, min_duration],
        outputs = [filtered_by_users_data, date_type, precise_date, min_nb_locations, min_duration])
    

    # III) Change of date type between choosing weekdays or selecting a specific date associated with a time interval
    date_type.change(fn = change_date_type, inputs = date_type, outputs = [weekdays, precise_date, time_range])
    

    # IV) Filtering of the data based on the user's choices for the filters (except for the user ids selection)
    gr.on(triggers = [user_ids_list.change, date_type.change, weekdays.change, precise_date.change, time_range.change,
                      min_nb_locations.change, min_duration.change], fn = filter_data,
                      inputs = [filtered_by_users_data, date_type, weekdays, precise_date, time_range,
                                min_nb_locations, min_duration],
                      outputs = [final_data, precise_date])
    

    # V) Displaying of the trajectories map (each time where the user modifies a filter)
    final_data.change(fn = display_map, inputs = [final_data, date_type, dataset_type], outputs = trajectories_map)

    
demo.launch(inbrowser=True)