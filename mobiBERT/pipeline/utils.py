"""
This Python module contains variables, functions, and other elements that may be useful for all modules of this library.
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

import os
from types import ModuleType
import inspect

# --------------------------------------------------------- VARIABLE DEFINITIONS -----------------------------------------------------------------------------------------------------------------------

_current_file = os.path.dirname(__file__)
"""Name of the directory containing the current file."""

library_data_dir = os.path.join(_current_file, "data")
"""Path to the directory containing this library's data."""

mlflow_data_dir = os.path.join(library_data_dir, "mlflow_data")
"""Path to the directory containing the MLflow tracking data."""

# --------------------------------------------------------- FUNCTION IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

def get_module_classes(module: ModuleType):
    """Returns a dictionary of all classes implemented in a specified module.

    The keys of this dictionary are the class names, and the values are the class objects.
    """

    all_classes = inspect.getmembers(module, inspect.isclass)
    # Keep only the classes that are defined within the specified module
    local_classes = {name: cls for name, cls in all_classes if cls.__module__ == module.__name__}
    return local_classes