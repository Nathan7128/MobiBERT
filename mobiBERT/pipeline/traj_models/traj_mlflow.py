"""
This Python module contains the implementation all MLflow elements associated with the training of our models :
    - Generic MLflow functions
    - Custom MLflow elements
    - etc.
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import packaging.version
import json
import os, platform
import subprocess, socket
import mlflow

import torch
from transformers.integrations.integration_utils import MLflowCallback
from transformers.utils import logging, ENV_VARS_TRUE_VALUES
from transformers.trainer import Trainer

# Internal libraries

import sys
sys.path.append("mobiBERT")

from pipeline.utils import mlflow_data_dir

# --------------------------------------------------------- FUNCTION IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

def _is_port_available(host = "127.0.0.1", port = 8080):
    """Check whether a specified port is available.

    Args:
        host (str, optional): Host address to launch the MLflow UI. Defaults to "127.0.0.1".
        port (int, optional): Port number to launch the MLflow UI. Defaults to 8080.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def kill_mlflow_ui(host = "127.0.0.1", port = "8080") :
    """Close all local servers associated with an MLflow UI based on a host adress and a port number.

    Args:
        host (str, optional): Host address to launch the MLflow UI. Defaults to "127.0.0.1".
        port (str, optional): Port number to launch the MLflow UI. Defaults to "8080".
    """
    # Attempt to run netstat to list active network connections and find the process ID (PID) bound to the target port
    if _is_port_available(host = host, port = int(port)):
        print("Error : this port is available and don't host an MLflow UI.\n")
        return
    
    result = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True)
    lines = result.decode().splitlines()

    # Iterate over each matching line to find a PID bound to the correct port
    for line in lines:
        parts = line.strip().split()
        # Ensure the line contains enough elements and matches the expected port
        if len(parts) >= 5 and (parts[1] == f"{host}:{port}" or parts[2] == f"{host}:{port}"):
            pid = int(parts[-1]) # PID is the last column
            try :
                system = platform.system()
                if system == "Windows":
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=True)
                else:  # macOS ou Linux
                    subprocess.run(["kill", "-9", str(pid)], check=True)
            except :
                pass

    print("MLflow UI has been stopped.\n")


def run_mlflow_ui(host = "127.0.0.1", port = "8080"):
    """Launch the MLflow UI on a local adress.

    Args:
        host (str, optional): Host address to launch the MLflow UI. Defaults to "127.0.0.1".
        port (str, optional): Port number to launch the MLflow UI. Defaults to "8080".
    """
    print("\nLAUNCHING THE MLFLOW UI...\n")
    
    if not _is_port_available(host = host, port = int(port)):
        print(f"Error : this port {port} is already in use.\n" +
            "Terminating the server currently running at this adress...\n")
        kill_mlflow_ui(host = host, port = port)
    
    subprocess.Popen([
        "mlflow", "server",
        "--host", host,
        "--port", port,
        "--backend-store-uri", f"file:/{mlflow_data_dir}"
    ])
    print(f"PLEASE ACCESS THE MLFLOW UI AT: http://{host}:{port}\n")

# --------------------------------------------------------- CLASS IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

class CustomMLflowCallbacks(MLflowCallback) :
    """Re-implementation of the Hugging Face MLflow Callback, allowing the adaptation
    of MLflow tracking for training parameters and metrics.

    The tracking parameters are logged at the beggining of training using the "setup" method.  
    Inverserly, the metrics are logged throughout the training using the "on_log" method.

    Args:
        params_to_log (dict[str]): Dictionnary of training parameters to track.
        tracking_metrics (_type_, optional): List of model evaluation metric names to track.  
            These names are specific to Hugging Face and must match those used by the Trainer.  
            Defaults to list[str].
    """

    def __init__(self, params_to_log : dict[str], tracking_metrics = list[str]):
        super().__init__()
        self._logger = logging.get_logger(__name__)
        self._params_to_log = params_to_log
        self._tracking_metrics = tracking_metrics

    def setup(self, args, state, model):
        self._log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._nested_run = os.getenv("MLFLOW_NESTED_RUN", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
        self._experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
        self._flatten_params = os.getenv("MLFLOW_FLATTEN_PARAMS", "FALSE").upper() in ENV_VARS_TRUE_VALUES
        self._run_id = os.getenv("MLFLOW_RUN_ID", None)
        self._max_log_params = os.getenv("MLFLOW_MAX_LOG_PARAMS", None)

        self._async_log = packaging.version.parse(self._ml_flow.__version__) >= packaging.version.parse("2.8.0")

        self._logger.debug(
            f"MLflow experiment_name={self._experiment_name}, run_name={args.run_name}, nested={self._nested_run},"
            f" tracking_uri={self._tracking_uri}"
        )
        if state.is_world_process_zero:
            if not self._ml_flow.is_tracking_uri_set():
                if self._tracking_uri:
                    self._ml_flow.set_tracking_uri(self._tracking_uri)
                    self._logger.debug(f"MLflow tracking URI is set to {self._tracking_uri}")
                else:
                    self._logger.debug(
                        "Environment variable `MLFLOW_TRACKING_URI` is not provided and therefore will not be"
                        " explicitly set."
                    )
            else:
                self._logger.debug(f"MLflow tracking URI is set to {self._ml_flow.get_tracking_uri()}")

            if self._ml_flow.active_run() is None or self._nested_run or self._run_id:
                if self._experiment_name:
                    # Use of set_experiment() ensure that Experiment is created if not exists
                    self._ml_flow.set_experiment(self._experiment_name)
                self._ml_flow.start_run(run_name=args.run_name, nested=self._nested_run)
                self._logger.debug(f"MLflow run started with run_id={self._ml_flow.active_run().info.run_id}")
                self._auto_end_run = True

            params_to_log = self._params_to_log
            if self._max_log_params and self._max_log_params.isdigit():
                max_log_params = int(self._max_log_params)
                if max_log_params < len(params_to_log):
                    self._logger.debug(
                        f"Reducing the number of parameters to log from {len(params_to_log)} to {max_log_params}."
                    )
                    params_to_log = params_to_log[:max_log_params]
            if self._async_log:
                self._ml_flow.log_params(
                    params_to_log, synchronous=False
                )
            else:
                self._ml_flow.log_params(params_to_log)
            mlflow_tags = os.getenv("MLFLOW_TAGS", None)
            if mlflow_tags:
                mlflow_tags = json.loads(mlflow_tags)
                self._ml_flow.set_tags(mlflow_tags)
        self._initialized = True



    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            metrics = {}
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    metrics[k] = v.item()
                else:
                    self._logger.warning(
                        f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                        "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                    )

            metrics = {k: v for k, v in metrics.items() if k in self._tracking_metrics}

            if self._async_log:
                self._ml_flow.log_metrics(metrics=metrics, step=state.global_step, synchronous=False)
            else:
                self._ml_flow.log_metrics(metrics=metrics, step=state.global_step)



class MLflowTracker(): 
    """Class that allows tracking of model training using the MLflow UI.

    An MLflowTracker is necessarily associated with a TrajModel and, in particular, with its Hugging Face Trainer.  
    In fact, if MLflow is used to track the model's training, the Trainer must be updated to enable this tracking.  

    To use this class, you must first create an MLflowTracker by specifying the name and the experiment associated
    with the training tracking of your model.  
    After that, you must set up the MLflowTracker by assigning a Hugging Face Trainer to it-adapted for MLflow-
    and providing additional information for the tracking such as the parameters to log in the UI, the metrics used to
    evaluate the model during training, etc.  
    Then, you will be ready to start training your model using the "train" method.

    The MLflow UI may be divided into several "experiments" (e.g., pre-trained models, fine-tuned models, etc.).  
    Each experiment includes several "runs", which correspond to tracked model training sessions.  
    A tracking consists of several sections, such as:
    - Details (creation date, duration, etc.)
    - A list of user-defined parameters, typically specific to the model
    - The progression of various metrics evaluated during the model's training

    Args:
        tracking_name (str): Name of the run as displayed in the MLflow experiment.
        experiment_name (str): Name of the experiment associated with this run.
    """

    def __init__(self, tracking_name: str, experiment_name: str):
        # Name of the tracking/run
        self._tracking_name = tracking_name
        self._experiment_name = experiment_name
        # Hugging Face Trainer that must be specified using the "setup" method
        self._trainer: Trainer = None
        # Boolean indicating whether system performance should be tracked during training
        self._system_metrics: bool = None


    def setup(self, trainer: Trainer, params_to_log: dict[str] = {}, system_metrics=True, tracking_metrics: list[str] = []) :
        """Sets up and configures the MLflowTracker by assigning it several pieces ofinformation about the model to track.

        Args:
            trainer (Trainer): Hugging Face Trainer associated with the model to track.
            params_to_log (dict[str], optional): Set of parameters to display in the MLflow UI and specifics to this
                model training. Defaults to {}.
            system_metrics (bool, optional): Whether to track your device's system performance during training
                (e.g., CPU, memory, etc.). Defaults to True.
            tracking_metrics (list[str], optional): List of metric names to track and that are computed by the
                Hugging Face Trainer during training. Defaults to [].
        """

        self._system_metrics = system_metrics
        # Create a Hugging Face Callback to link MLflow and Hugging Face
        callback = CustomMLflowCallbacks(params_to_log=params_to_log, tracking_metrics=tracking_metrics)
        trainer.remove_callback(MLflowCallback)
        trainer.add_callback(callback)
        self._trainer = trainer


    def train(self):
        """Starts the tracking and the training of your model.

        Note: This method does not launch the MLflow UI.  
        You must use the "run_mlflow_ui" function from this module.
        """

        if self._trainer is None:
            raise ValueError("Error : MLflow tracking is not set up for this model!\n"
                             "Use the 'setup' method!")
        
        # Specify the directory where the data assiociated with this tracking will be stored
        mlflow.set_tracking_uri(uri=f"file:/{mlflow_data_dir}")
        # Specify the experiment associated with this run
        mlflow.set_experiment(experiment_name=self._experiment_name)

        # Start the run
        with mlflow.start_run(run_name=self._tracking_name, log_system_metrics=self._system_metrics):
            # Start the model training
            training = self._trainer.train()
            
            # Once training is complete, add additional system metrics about the training runtime
            if self._system_metrics :
                mlflow.log_metrics({
                    "train_runtime" : training.metrics["train_runtime"],
                    "epoch_runtime" : training.metrics["train_runtime"]/training.metrics["epoch"],
                })