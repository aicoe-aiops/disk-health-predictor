import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from .._types import DevSmartT
from ._base import DiskHealthClassifier

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

    # json with vendor names as keys
    # and features used for prediction as values
    CONFIG_FILE = "config.json"
    PREDICTION_CLASSES = {-1: "Unknown", 0: "Good", 1: "Warning", 2: "Bad"}

    # model name prefixes to identify vendor
    VENDOR_MODELNAME_PREFIXES = {
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

        # ensure all vendors whose context is defined in config file
        # have models and scalers saved inside model_dirpath
        for vendor in self.model_context:
            scaler_path = os.path.join(model_dirpath, vendor + "_scaler.pkl")
            if not os.path.isfile(scaler_path):
                raise Exception(f"Missing scaler file: {scaler_path}")
            model_path = os.path.join(model_dirpath, vendor + "_predictor.pkl")
            if not os.path.isfile(model_path):
                raise Exception(f"Missing model file: {model_path}")

        self.model_dirpath = model_dirpath

    def __preprocess(
        self, disk_days: Sequence[DevSmartT], vendor: str
    ) -> Optional[np.ndarray]:
        """Scales and transforms input dataframe to feed it to prediction model
        Arguments:
            disk_days {dict} -- dict where key is date, value is smartctl data extracted
                                from that date e.g. {"2020-01-31 12:12:01": {"wwn":
                                {"naa": 5, "oui": 3152}, "ata_smart_attributes":
                                {"table": [{"id": 1, "raw": {"value": 60404968, ... }
            vendor {str} -- vendor of the hard disk
        Returns:
            numpy.ndarray -- (n, d) shaped array of n days worth of data and d
                                features, scaled
        """
        # get the attributes that were used to train model for current vendor
        try:
            model_smart_attr = self.model_context[vendor]
        except KeyError:
            RHDiskHealthClassifier.LOGGER.debug(
                "No context (SMART attributes on which model has been trained) found \
                    for vendor: {}".format(
                    vendor
                )
            )
            return None

        # flattened dict {"smart_1_raw": 1, "user_capacity": 1000, "smart_9_norm": 12}
        disk_days_smartctl_attrs = self.extract_ata_smart_attrs(disk_days)

        # convert to structured array, keeping only the required features
        # assumes all data is in float64 dtype
        try:
            struc_dtypes = [(attr, np.float64) for attr in model_smart_attr]
            values = [
                tuple(daydata[attr] for attr in model_smart_attr)
                for _, daydata in disk_days_smartctl_attrs.items()
            ]
            disk_days_sa = np.array(values, dtype=struc_dtypes)
        except KeyError:
            RHDiskHealthClassifier.LOGGER.debug(
                "Mismatch in SMART attributes used to train model and SMART attributes\
                     available"
            )
            return None

        # do not include capacity_bytes in this. only use smart_attrs
        disk_days_smartctl_attrs_sa = disk_days_sa[
            [attr for attr in model_smart_attr if "smart_" in attr]
        ]

        # view structured array as 2d array for applying rolling window transforms
        # NOTE: this is new behavior from numpy 1.15 to 1.16
        disk_days_smartctl_attrs = structured_to_unstructured(
            disk_days_smartctl_attrs_sa
        )

        # featurize n (6 to 12) days data - mean,std,coefficient of variation
        # current model is trained on 6 days of data because that is what will be
        # available at runtime

        # rolling time window interval size in days
        roll_window_size = 6

        # number of days of data available
        dataset_size = disk_days_smartctl_attrs.shape[0] - roll_window_size + 1

        # rolling means
        means = np.vstack(
            [
                disk_days_smartctl_attrs[i : i + roll_window_size, ...].mean(axis=0)
                for i in range(dataset_size)
            ]
        )

        # rolling stds
        stds = np.vstack(
            [
                disk_days_smartctl_attrs[i : i + roll_window_size, ...].std(
                    axis=0, ddof=1
                )
                for i in range(dataset_size)
            ]
        )

        # coefficient of variation
        # there might be cases where mean for a smart attribute is 0
        # this is not necessarily something to warn about
        with np.errstate(divide="ignore", invalid="ignore"):
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
        scaler_path = os.path.join(self.model_dirpath, vendor + "_scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        featurized = scaler.transform(featurized)
        return featurized

    @staticmethod
    def _estimate_vendor_from_modelname(model_name: str) -> Optional[str]:
        """Returns the vendor name for a given hard disk model name
        Arguments:
            model_name {str} -- hard disk model name
        Returns:
            str -- vendor name
        """
        for (
            prefix,
            vendor,
        ) in RHDiskHealthClassifier.VENDOR_MODELNAME_PREFIXES.items():
            if model_name.startswith(prefix):
                return vendor.lower()
        # print error message
        RHDiskHealthClassifier.LOGGER.debug(
            f"Could not infer vendor from model name {model_name}"
        )
        return None

    @staticmethod
    def _extract_vendor_from_smartctl(disk_days: dict) -> Optional[str]:
        """Returns the vendor name for the hard disk given smartctl data
        Arguments:
            disk_days {dict} -- dictionary of smartctl data. each key is
            a day, value is smartctl data collected by ceph
        Returns:
            str -- vendor name
        """
        # get vendor preferably as a smartctl attribute
        # check if the "vendor" key exists in data collected from any day
        vendor_key_found = any(
            "vendor" in daydata.keys() for _, daydata in disk_days.items()
        )

        if vendor_key_found:
            # sanity check
            # there should only be one unique value for vendor for all days
            extracted_vendors = set(
                [daydata["vendor"] for _, daydata in disk_days.items()]
            )
            if len(extracted_vendors) > 1:
                RHDiskHealthClassifier.LOGGER.warning(
                    f'Multiple values found for "vendor": {extracted_vendors}. \
                        Will randomly pick one of these.'
                )

            vendor = list(extracted_vendors)[0]

        # if not available as smartctl attr then infer using model name
        else:
            RHDiskHealthClassifier.LOGGER.debug(
                '"vendor" field not found in smartctl output. Will try to infer \
                    vendor from model name.'
            )

            # check if the "model_name" key exists in data collected from any day
            modelname_key_found = any(
                "model_name" in daydata.keys() for _, daydata in disk_days.items()
            )
            if modelname_key_found:
                # sanity check
                # there should only be one unique value for vendor for all days
                extracted_modelnames = set(
                    [daydata["model_name"] for _, daydata in disk_days.items()]
                )
                if len(extracted_modelnames) > 1:
                    RHDiskHealthClassifier.LOGGER.warning(
                        f'Multiple values found for "model_name": {extracted_modelnames}. \
                            Will randomly pick one of these.'
                    )

                model_name = list(extracted_modelnames)[0]
                vendor = RHDiskHealthClassifier._estimate_vendor_from_modelname(
                    model_name
                )
            else:
                # if model name was also not found, there is no other way to get vendor
                RHDiskHealthClassifier.LOGGER.warning(
                    "Model name not found in smartctl data."
                )
                vendor = None

        return vendor

    def predict(self, disk_days: Sequence[DevSmartT]) -> str:
        # get vendor
        vendor = RHDiskHealthClassifier._extract_vendor_from_smartctl(disk_days)

        # print error message, return Unknown, and continue execution
        if vendor is None:
            RHDiskHealthClassifier.LOGGER.debug(
                "vendor could not be determiend. This may be because \
                DiskPredictor has never encountered this vendor before, \
                    or the model name is not according to the vendor's \
                        naming conventions known to DiskPredictor"
            )
            return RHDiskHealthClassifier.PREDICTION_CLASSES[-1]

        # preprocess for feeding to model
        preprocessed_data = self.__preprocess(disk_days, vendor)
        if preprocessed_data is None:
            return RHDiskHealthClassifier.PREDICTION_CLASSES[-1]

        # get model for current vendor
        model_path = os.path.join(self.model_dirpath, vendor + "_predictor.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # use prediction for most recent day
        # TODO: ensure that most recent day is last element and most previous day
        # is first element in input disk_days
        pred_class_id = model.predict(preprocessed_data)[-1]
        return RHDiskHealthClassifier.PREDICTION_CLASSES[pred_class_id]
