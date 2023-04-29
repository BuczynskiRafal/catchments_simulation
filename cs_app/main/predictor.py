import pickle
import swmmio
from tensorflow.keras.models import load_model 



# insert scaler 
with open(r"C:\Users\Dell\Documents\Git\catchments_simulation\cs_app\swmm_model\scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)


# insert ann model
model = load_model(r"C:\Users\Dell\Documents\Git\catchments_simulation\cs_app\swmm_model")


def predict_runoff(swmmio_model: swmmio.Model):
    """Predict runoff using a machine learning model.

    Args:
        swmmio_model (swmmio.Model): A SWMM model object with subcatchment data.

    Returns:
        np.ndarray: An array of predicted runoff values for each subcatchment.

    Example:
        >>> swmm_model = swmmio.Model("example.inp")
        >>> predict_runoff(swmm_model)
        array([0.1, 0.2, 0.3, 0.4, 0.5])
    """
    data = swmmio_model.subcatchments.dataframe[["PercImperv", "Width", "PercSlope", "N-Imperv", "N-Perv", "S-Imperv", "S-Perv", "PctZero"]]
    data = loaded_scaler.transform(data)
    return model.predict(data).flatten()


