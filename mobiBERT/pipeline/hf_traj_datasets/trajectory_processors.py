"""
This Python module contains the implementation of all functions used for processing raw trajectory data:
  - Trajectory simplification
  - Coordinates encoding
  - And more...
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import pandas as pd
import geohash2
import concurrent.futures
from tqdm import tqdm

# --------------------------------------------------------- FUNCTION IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

def round_timestamp(dataframe: pd.DataFrame, timestamp_rounding=120):
    """Simplify the locations for each user: this allows removing locations that are temporaly close (based on a selected time precision)
    in order to reduce the number of locations and thus the amount of data.
    
    These locations, being temporally close,  are considered "repetitive", which is why we group them.

    Args:
        dataframe (pd.DataFrame): Dataframe that contains trajectory data.
        timestamp_rounding (int, optional): Precision (in seconds) used to round the timestamp. Defaults to 120.
    """

    simplified_dataframe = dataframe.copy()

    # Round the timestamp
    simplified_dataframe["ts"] = (dataframe["ts"] // timestamp_rounding).astype("int64")
    simplified_dataframe["ts"] *= timestamp_rounding

    # Aggregate/merge locations with the same rounded timestamp by attributing them their average longitude and latitude
    simplified_dataframe = simplified_dataframe.groupby(["uid", "ts"]).agg({"latitude": "mean", "longitude": "mean"}).reset_index()

    return simplified_dataframe
        


def geohash_encode(dataframe : pd.DataFrame, encoding_precision=8):
    """Encode coordinated (longitude and latitude) associated with each locations using GeoHash encoding.

    This function add a "geohash" column that contains encoded coordinates to the dataframe passed as a parameter.

    Args:
        dataframe (pd.DataFrame): Dataframe that contains trajectory data.
        encoding_precision (int, optional): Precision used to encode coordinates with the GeoHash encoder. Defaults to 8.
    """

    encoded_dataframe = dataframe.copy()

    encoded_dataframe["geohash"] = dataframe.apply(lambda x: geohash2.encode(x["latitude"], x["longitude"],
                                                                             precision=encoding_precision), axis=1)

    return encoded_dataframe



def _create_user_sequences(user_trajectories: pd.DataFrame, temporal_window=3*60*60, temporal_lag=20*60):
    """Create and process sequences of encoded locations (trajectories) for a single user.

    This method is based on the "sliding window" technique, which is a principle of self-supervised learning.

    Args:
        user_trajectories (pd.DataFrame): Trajectory data of a unique user.
        temporal_window (_type_, optional): Sliding temporal window use (= maximum duration in seconds of each sequence). Defaults to 3*60*60.
        temporal_lag (_type_, optional): Minimum temporal lag between the start timestamps of each sequence. Defaults to 20*60.

    Returns:
        list[tuple[int, pd.DataFrame]]: List of tuples, where each tuple contains a sequence and the corresponding user id.
    """

    # Id of the user for these trajectories
    user_id = user_trajectories["uid"].iloc[0]

    sorted_trajectories = user_trajectories.sort_values(by="ts")

    # Minimum and maximum timestamp for this user
    min_timestamp = sorted_trajectories["ts"].min()
    max_timestamp = sorted_trajectories["ts"].max()

    # List that will contain all the sequences created using the sliding window technique
    user_sequences: list[tuple[int, pd.DataFrame]] = []

    # Loop that iterates over the start timestamp of each sequence
    for sequence_start in range(min_timestamp, max_timestamp - temporal_window, temporal_lag):
        # End timestamp of the current sequence being iterated
        end_sequence = sequence_start + temporal_window

        # Filter for the sequence corresponding to the temporal window
        sequence_mask = (sorted_trajectories["ts"] >= sequence_start) & (sorted_trajectories["ts"] < end_sequence)
        sequence = sorted_trajectories[sequence_mask]

        # Add the created sequence to the list containing all sequences for this user
        if len(sequence) > 0:
            user_sequences.append((user_id, sequence))

    return user_sequences



def create_sequences(dataframe: pd.DataFrame, temporal_window=3*60*60, temporal_lag=20*60, use_threading=False):
    """Create a Pandas dataframe of spatio-temporal sequences (trajectories) using the "sliding window" technique,
    which is a principle of self-supervised learning.

    The created dataframe will contain, for each sequence, the following information (columns):
    - "label": user id associated with the sequence
    - "text": list of encoded coordinates (longitude and latitude) composing the sequence
    - "timestamps": list of timestamps (in seconds) associated with each location (encoded coordinates) in the sequence

    Args:
        dataframe (pd.DataFrame): Dataframe of trajectory data.
        temporal_window (_type_, optional): Sliding temporal window use (= maximum duration in seconds of each sequence). Defaults to 3*60*60.
        temporal_lag (_type_, optional): Minimum temporal lag between the start timestamps of each sequence. Defaults to 20*60.
        use_threading (bool, optional): Whether to use multithreading to create and process all the user trajectories in parallel. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe of spatio-temporal sequences, ready to be transformed into a Hugging Face dataset.
    """

    # 1) Create and process sequences for all users

    # List that will contain all sequences for all users
    all_sequences = []

    # a) Create sequences using multithreading
    if use_threading:
        # Create the threads pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # List that will contain all tasks to be executed in parallel
            tasks_list = []
            # Add all tasks to the list (one task per user)
            for user_id, user_trajectories in dataframe.groupby("uid"):
                task = executor.submit(_create_user_sequences, user_trajectories, temporal_window, temporal_lag)
                tasks_list.append(task)

            # Execute the tasks using multithreading
            for future in tqdm(concurrent.futures.as_completed(tasks_list), total=len(tasks_list), desc="Creating sequences"):
                user_sequences = future.result()
                # Add the sequences (if they are not empty) created for the iterated user to the list of all sequences
                all_sequences.extend(user_sequences)

    # b) Create sequences without parallel processing
    else:
        # Create the sequences for all users
        for user_id, user_trajectories in tqdm(dataframe.groupby("uid"), total=dataframe["uid"].nunique(), desc="Creating sequences"):
            # Sequences created for the current user
            user_sequences = _create_user_sequences(user_trajectories=user_trajectories)
            all_sequences.extend(user_sequences)


    # 2) Creation of the dataframe that will be ready to be convert into a Hugging Face dataset

    sequence_dict_list = []

    for sequence in all_sequences:
        sequence[1].drop("uid", axis=1, inplace=True)
        sequence[1].reset_index(drop=True)
        sequence_dict_list.append({
            "label": sequence[0],
            "text": " ".join(sequence[1]["geohash"].to_list()),
            "timestamps": sequence[1]["ts"].to_list()
        })

    return pd.DataFrame(sequence_dict_list)