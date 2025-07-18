"""
This Python module contains the implementation of the TrajMetric class and its subclasses, i.e., all the specific metrics used to evaluate
the trajectory models (e.g., accuracy, macro F1 score, etc.).
"""

# --------------------------------------------------------- LIBRARY IMPORTS -----------------------------------------------------------------------------------------------------------------------

# External libraries

from abc import abstractmethod, ABC
import numpy as np

from transformers.trainer_utils import EvalPrediction
import evaluate

# --------------------------------------------------------- VARIABLE DEFINITIONS -----------------------------------------------------------------------------------------------------------------------

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# --------------------------------------------------------- CLASS IMPLEMENTATIONS -----------------------------------------------------------------------------------------------------------------------

class TrajMetric(ABC):
    """Base class for all specific metrics implemented (e.g., accuracy, macro F1 score, recall, etc.).

    Each metric is defined by its name (the "_name" attribute) and a function used to compute its value based on a
    model prediction (the "compute" method).

    This model prediction must be a Hugging Face EvalPrediction object, composed of predicted logits and true targets (labels)
    for the corresponding prediction.

    Args:
        metric_name (str): Name of the metric.
    """

    def __init__(self, metric_name: str):
        self._name = metric_name


    def get_name(self):
        return self._name


    @abstractmethod
    def compute(self, eval_pred: EvalPrediction):
        """Computes the value of the correponding metric based on a model prediction (an EvalPrediction object).

        Args:
            eval_pred (EvalPrediction): Prediction of a Hugging Face model for given inputs.  
                Each EvalPrediction object (for a classification task) consists of predicted logits
                and the true targets (labels) for the corresponding prediction.
        """
        pass



class TrajAccuracy(TrajMetric):
    """Class that inherits from TrajMetric and defines the accuracy metric.

    See the TrajMetric documentation for more information.
    """

    def __init__(self):
        super().__init__("Accuracy")


    def compute(self, eval_pred: EvalPrediction):
        logits_pred, labels = eval_pred
        predictions = np.argmax(logits_pred, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    


class TrajF1(TrajMetric):
    """Class that inherits from TrajMetric and defines the macro F1 score metric.

    See the TrajMetric documentation for more information.
    """

    def __init__(self):
        super().__init__("F1")

    
    def compute(self, eval_pred: EvalPrediction) :
        logits_pred, labels = eval_pred
        predictions = np.argmax(logits_pred, axis=-1)
        return f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    


class TrajPrecision(TrajMetric):
    """Class that inherits from TrajMetric and defines the precision metric.

    See the TrajMetric documentation for more information.
    """

    def __init__(self):
        super().__init__("Precision")

    
    def compute(self, eval_pred: EvalPrediction):
        logits_pred, labels = eval_pred
        predictions = np.argmax(logits_pred, axis=-1)
        return precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    


class TrajRecall(TrajMetric):
    """Class that inherits from TrajMetric and defines the recall metric.

    See the TrajMetric documentation for more information.
    """

    def __init__(self):
        super().__init__("Recall")

    
    def compute(self, eval_pred: EvalPrediction):
        logits_pred, labels = eval_pred
        predictions = np.argmax(logits_pred, axis=-1)
        return recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    


class TrajTop_K_Accuracy(TrajMetric):
    """Class that inherits from TrajMetric and defines the Top-k accuracy.

    See the TrajMetric documentation for more information.

    Args:
        top_k_length (int, optional): The number of top predicted values to consider.  
            A prediction is considered correct (True) if the target label is among the top-k predictions. Defaults to 5.
    """

    def __init__(self, top_k_length=5):
        super().__init__(f"Top_{top_k_length}_Accuracy")
        self._top_k_length = top_k_length

    
    def compute(self, eval_pred: EvalPrediction) :
        logits_pred, labels = eval_pred
        top_k_length = self._top_k_length
        # top_k_preds contains, for each sequence (each row), the input_ids corresponding to the top-k highest predicted logits
        top_k_preds = np.argsort(logits_pred, axis=1)[:, -top_k_length:]
        # zip allows us to create a list of n tuples, where n = min(len(references), (top_k_preds))
        # and each tuple contains the i-th elements of references & top_k_preds for i = 1, ..., n
        top_k_accuracy = np.mean([ref in top_k for ref, top_k in zip(labels, top_k_preds)])
        return top_k_accuracy



class TrajTop_K_F1(TrajMetric):
    """Class that inherits from TrajMetric and defines the Top-k macro F1 score.

    See the TrajMetric documentation for more information.

    Args:
        top_k_length (int, optional): The number of top predicted values to consider.  
            A prediction is considered correct (True) if the target label is among the top-k predictions. Defaults to 5.
    """

    def __init__(self, top_k_length=5):
        super().__init__(f"Top_{top_k_length}_F1")
        self._top_k_length = top_k_length

    
    def compute(self, eval_pred: EvalPrediction):
        logits_pred, labels = eval_pred
        top_k_length = self._top_k_length
        # Get indices of top-k predictions using argpartition for efficiency
        top_k_preds = np.argpartition(logits_pred, -top_k_length, axis=1)[:, -top_k_length:]
        # top_k_preds contains, for each row, the top-k labels with the highest predicted logit
        # -> for each row, the first column corresponds to the label with the k-st highest predicted logit,
        # the second column to the (k - 1)-st highest, ..., and the last column to the highest predicted logit.
        
        # Compute top-k F1 score
        f1_scores = []
        for ref, top_k in zip(labels, top_k_preds):
            # ref : true label of the i-st row (sequence)
            # top_k : top-k predicted labels for the i-th row, sorted in ascending order of logits
            # Ensure ref is treated as a set of indices
            ref_set = set(ref) if isinstance(ref, (list, set, np.ndarray)) else {ref}
            top_k_set = set(top_k)
            
            if len(ref_set) == 0 and len(top_k_set) == 0:
                f1_scores.append(1.0)  # Perfect match when both are empty
            elif len(ref_set) == 0 or len(top_k_set) == 0:
                f1_scores.append(0.0)  # F1 is 0 if either is empty
            else:
                intersection = len(ref_set.intersection(top_k_set))
                f1 = 2 * intersection / (len(ref_set) + len(top_k_set))
                f1_scores.append(f1)
        
        top_k_f1 = np.mean(f1_scores)
        return top_k_f1