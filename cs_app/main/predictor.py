"""
This module contains functions to predict runoff using a machine learning model.

The model and scaler are loaded from the 'swmm_model' directory.
The predict_runoff function takes a SWMM model object as input and returns an array of
predicted runoff values for each subcatchment.
"""
import os
import swmmio
from tensorflow.keras.models import load_model


current_directory = os.path.dirname(os.path.abspath(__file__))
scaler_file_path = os.path.join(current_directory, "..", "swmm_model", "scaler.pkl")
model_path = os.path.join(current_directory, "..", "swmm_model")

try:
    predict_model = load_model(model_path)
except FileNotFoundError:
    print(f"Cannot load model: {model_path}")
    raise


def predict_runoff(swmmio_model: swmmio.Model):
    """
    Predict runoff using a machine learning model.

    Parameters
    ----------
    swmmio_model : swmmio.Model
        A SWMM model object with subcatchment data.

    Returns
    -------
    np.ndarray
        An array of predicted runoff values for each subcatchment.

    Example
    -------
    >>> swmm_model = swmmio.Model("example.inp")
    >>> predict_runoff(swmm_model)
    array([0.1, 0.2, 0.3, 0.4, 0.5])
    """
    data = swmmio_model.subcatchments.dataframe[
        [
            "PercImperv",
            "Width",
            "PercSlope",
            "N-Imperv",
            "N-Perv",
            "S-Imperv",
            "S-Perv",
            "PctZero",
        ]
    ]
    return predict_model.predict(data).flatten()
