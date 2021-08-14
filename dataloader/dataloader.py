from dataloader.dataset import MEDataset
import numpy as np
import pandas as pd
from torch.utils.data import (
    DataLoader
)


def LOSO_specific_subject_sequence_generate(data: pd.DataFrame, sub_column: str,
                                            sub: str) -> tuple:
    """Generate train and test data using leave-one-subject-out for
    specific subject

    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame
    sub_column : str
        Subject column in DataFrame
    sub : str
        Subject to be leave out in training

    Returns
    -------
    tuple
        Return training and testing DataFrame
    """

    # Mask for the training
    mask = data["Subject"].isin([sub])

    # Masking for the specific data
    train_data = data[~mask]
    test_data = data[mask]

    return train_data, test_data


def LOSO_sequence_generate(data: pd.DataFrame, sub_column: str) -> tuple:
    """Generate train and test data using leave-one-subject-out for
    specific subject

    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame
    sub_column : str
        Subject column in DataFrame

    Returns
    -------
    tuple
        Return training and testing list DataFrame
    """
    # Save the training and testing list for all subject
    train_list = []
    test_list = []

    # Unique subject in `sub_column`
    subjects = np.unique(data[sub_column])

    for subject in subjects:
        # Mask for the training
        mask = data["Subject"].isin([subject])

        # Masking for the specific data
        train_data = data[~mask].reset_index(drop=True)
        test_data = data[mask].reset_index(drop=True)

        train_list.append(train_data)
        test_list.append(test_data)

    return train_list, test_list


def get_loader(csv_file, label_mapping,
               image_root, batch_size,
               catego, device, train=True,
               shuffle=True):
    dataset = MEDataset(data_info=csv_file,
                        label_mapping=label_mapping,
                        image_root=image_root,
                        catego=catego,
                        device=device,
                        train=train)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True)

    return dataset, dataloader