import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np

from .._types import DevSmartT
from ._base import DiskHealthClassifier


def get_diskfailurepredictor_path() -> str:
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    return dir_path


# # DUMMY DEFINITION FOR TESTING PURPOSES
# class RHDiskHealthClassifier:
#     def __init__(self) -> None:
#         pass
#     def predict(self):
#         return "Warning"


class RHDiskHealthClassifier(DiskHealthClassifier):
    """Disk failure prediction module developed at Red Hat
    This class implements a disk failure prediction module.
    """

    # json with manufacturer names as keys
    # and features used for prediction as values
    CONFIG_FILE = "config.json"
    PREDICTION_CLASSES = {-1: "Unknown", 0: "Good", 1: "Warning", 2: "Bad"}

    # model name prefixes to identify vendor
    MANUFACTURER_MODELNAME_PREFIXES = {
        "WDC": "WDC",
        "Toshiba": "Toshiba",  # for cases like "Toshiba xxx"
        "TOSHIBA": "Toshiba",  # for cases like "TOSHIBA xxx"
        "toshiba": "Toshiba",  # for cases like "toshiba xxx"
        "S": "Seagate",  # for cases like "STxxxx" and "Seagate BarraCuda ZAxxx"
        "ZA": "Seagate",  # for cases like "ZAxxxx"
        "Hitachi": "Hitachi",
        "HGST": "HGST",
    }

    LOGGER = logging.getLogger()

    def __init__(self) -> None:
        """
        This function may throw exception due to wrong file operation.
        """
        self.model_dirpath = ""
        self.model_context: Dict[str, List[str]] = {}

    def initialize(self, model_dirpath: str) -> None:
        """Initialize all models. Save paths of all trained model files to list
        Arguments:
            model_dirpath {str} -- path to directory of trained models
        Returns:
            str -- Error message. If all goes well, return None
        """
        # read config file as json, if it exists
        config_path = os.path.join(model_dirpath, self.CONFIG_FILE)
        if not os.path.isfile(config_path):
            raise Exception("Missing config file: " + config_path)
        with open(config_path) as f_conf:
            self.model_context = json.load(f_conf)

        # ensure all manufacturers whose context is defined in config file
        # have models and scalers saved inside model_dirpath
        for manufacturer in self.model_context:
            scaler_path = os.path.join(model_dirpath, manufacturer + "_scaler.pkl")
            if not os.path.isfile(scaler_path):
                raise Exception(f"Missing scaler file: {scaler_path}")
            model_path = os.path.join(model_dirpath, manufacturer + "_predictor.pkl")
            if not os.path.isfile(model_path):
                raise Exception(f"Missing model file: {model_path}")

        self.model_dirpath = model_dirpath

    def __preprocess(
        self, disk_days: Sequence[DevSmartT], manufacturer: str
    ) -> Optional[np.ndarray]:
        """Scales and transforms input dataframe to feed it to prediction model
        Arguments:
            disk_days {list} -- list in which each element is a dictionary with key,val
                                as feature name,value respectively.
                                e.g.[{'smart_1_raw': 0, 'user_capacity': 512 ...}, ...]
            manufacturer {str} -- manufacturer of the hard drive
        Returns:
            numpy.ndarray -- (n, d) shaped array of n days worth of data and d
                                features, scaled
        """
        # get the attributes that were used to train model for current manufacturer
        try:
            model_smart_attr = self.model_context[manufacturer]
        except KeyError:
            RHDiskHealthClassifier.LOGGER.debug(
                "No context (SMART attributes on which model has been trained) found \
                    for manufacturer: {}".format(
                    manufacturer
                )
            )
            return None

        # convert to structured array, keeping only the required features
        # assumes all data is in float64 dtype
        try:
            struc_dtypes = [(attr, np.float64) for attr in model_smart_attr]
            values = [
                tuple(day[attr] for attr in model_smart_attr) for day in disk_days
            ]
            disk_days_sa = np.array(values, dtype=struc_dtypes)
        except KeyError:
            RHDiskHealthClassifier.LOGGER.debug(
                "Mismatch in SMART attributes used to train model and SMART attributes\
                     available"
            )
            return None

        # view structured array as 2d array for applying rolling window transforms
        # do not include capacity_bytes in this. only use smart_attrs
        disk_days_attrs = (
            disk_days_sa[[attr for attr in model_smart_attr if "smart_" in attr]]
            .view(np.float64)
            .reshape(disk_days_sa.shape + (-1,))
        )

        # featurize n (6 to 12) days data - mean,std,coefficient of variation
        # current model is trained on 6 days of data because that is what will be
        # available at runtime

        # rolling time window interval size in days
        roll_window_size = 6

        # rolling means generator
        dataset_size = disk_days_attrs.shape[0] - roll_window_size + 1
        gen = (
            disk_days_attrs[i : i + roll_window_size, ...].mean(axis=0)
            for i in range(dataset_size)
        )
        means = np.vstack(gen)

        # rolling stds generator
        gen = (
            disk_days_attrs[i : i + roll_window_size, ...].std(axis=0, ddof=1)
            for i in range(dataset_size)
        )
        stds = np.vstack(gen)

        # coefficient of variation
        cvs = stds / means
        cvs[np.isnan(cvs)] = 0
        featurized = np.hstack(
            (
                means,
                stds,
                cvs,
                disk_days_sa["user_capacity"][:dataset_size].reshape(-1, 1),
            )
        )

        # scale features
        scaler_path = os.path.join(self.model_dirpath, manufacturer + "_scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        featurized = scaler.transform(featurized)
        return featurized

    @staticmethod
    def __get_manufacturer(model_name: str) -> Optional[str]:
        """Returns the manufacturer name for a given hard drive model name
        Arguments:
            model_name {str} -- hard drive model name
        Returns:
            str -- manufacturer name
        """
        for (
            prefix,
            manufacturer,
        ) in RHDiskHealthClassifier.MANUFACTURER_MODELNAME_PREFIXES.items():
            if model_name.startswith(prefix):
                return manufacturer.lower()
        # print error message
        RHDiskHealthClassifier.LOGGER.debug(
            f"Could not infer manufacturer from model name {model_name}"
        )
        return None

    def predict(self, disk_days: Sequence[DevSmartT]) -> str:
        # get manufacturer preferably as a smartctl attribute
        # if not available then infer using model name
        manufacturer = disk_days[0].get("vendor")
        if manufacturer is None:
            RHDiskHealthClassifier.LOGGER.debug(
                '"vendor" field not found in smartctl output. Will try to infer \
                    manufacturer from model name.'
            )
            manufacturer = RHDiskHealthClassifier.__get_manufacturer(
                disk_days[0].get("model_name", "")
            )

        # print error message, return Unknown, and continue execution
        if manufacturer is None:
            RHDiskHealthClassifier.LOGGER.debug(
                "Manufacturer could not be determiend. This may be because \
                DiskPredictor has never encountered this manufacturer before, \
                    or the model name is not according to the manufacturer's \
                        naming conventions known to DiskPredictor"
            )
            return RHDiskHealthClassifier.PREDICTION_CLASSES[-1]

        # preprocess for feeding to model
        preprocessed_data = self.__preprocess(disk_days, manufacturer)
        if preprocessed_data is None:
            return RHDiskHealthClassifier.PREDICTION_CLASSES[-1]

        # get model for current manufacturer
        model_path = os.path.join(self.model_dirpath, manufacturer + "_predictor.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # use prediction for most recent day
        # TODO: ensure that most recent day is last element and most previous day
        # is first element in input disk_days
        pred_class_id = model.predict(preprocessed_data)[-1]
        return RHDiskHealthClassifier.PREDICTION_CLASSES[pred_class_id]
