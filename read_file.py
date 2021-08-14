import numpy as np
import pandas as pd


def read_csv(path: str) -> tuple:
    """Read in the csv from given path,
    and return the label mapping

    Parameters
    ----------
    path : str
        Path for CSV file

    Returns
    -------
    Tuple
        Return data and label mapping
    """
    # Read in dataset
    data = pd.read_csv(path,
                       dtype={"Subject": str})

    # Label the emotions into number
    label_mapping = {
        emotion: idx for idx, emotion in enumerate(np.unique(data.loc[:, "Estimated Emotion"]))
    }

    return data, label_mapping