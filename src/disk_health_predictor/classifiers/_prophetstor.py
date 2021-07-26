import json
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

from .._types import AttrDiffsT, AttrNamesT, DevSmartT
from ._base import DiskHealthClassifier

# # DUMMY DEFINITION FOR TESTING PURPOSES
# class PSDiskHealthClassifier:
#     def __init__(self) -> None:
#         pass

#     def predict(self):
#         return "Good"


class PSDiskHealthClassifier(DiskHealthClassifier):
    """Disk failure prediction developed at ProphetStor
    This class implements a disk failure prediction module.
    """

    CONFIG_FILE = "config.json"
    EXCLUDED_ATTRS = ["smart_9_raw", "smart_241_raw", "smart_242_raw"]

    def __init__(self) -> None:
        """
        This function may throw exception due to wrong file operation.
        """

        self.model_dirpath = ""
        self.model_context: Dict[str, List[str]] = {}

    def initialize(self, model_dirpath: str) -> None:
        """
        Initialize all models.
        Args: None
        Returns:
            Error message. If all goes well, return an empty string.
        Raises:
        """

        config_path = os.path.join(model_dirpath, self.CONFIG_FILE)
        if not os.path.isfile(config_path):
            raise Exception(f"Missing config file: {config_path}")
        with open(config_path) as f_conf:
            self.model_context = json.load(f_conf)

        for model_name in self.model_context:
            model_path = os.path.join(model_dirpath, model_name)

            if not os.path.isfile(model_path):
                raise Exception(f"Missing model file: {model_path}")

        self.model_dirpath = model_dirpath

    def __preprocess(self, disk_days: Sequence[DevSmartT]) -> Sequence[DevSmartT]:
        """
        Preprocess disk attributes.
        Args:
            disk_days: Refer to function predict(...).
        Returns:
            new_disk_days: Processed disk days.
        """

        req_attrs = []
        new_disk_days = []

        attr_list = set.intersection(*[set(disk_day.keys()) for disk_day in disk_days])
        for attr in attr_list:
            if (
                attr.startswith("smart_") and attr.endswith("_raw")
            ) and attr not in self.EXCLUDED_ATTRS:
                req_attrs.append(attr)

        for disk_day in disk_days:
            new_disk_day = {}
            for attr in req_attrs:
                if float(disk_day[attr]) >= 0.0:
                    new_disk_day[attr] = disk_day[attr]

            new_disk_days.append(new_disk_day)

        return new_disk_days

    @staticmethod
    def __get_diff_attrs(
        disk_days: Sequence[DevSmartT],
    ) -> Tuple[AttrNamesT, AttrDiffsT]:
        """
        Get 5 days differential attributes.
        Args:
            disk_days: Refer to function predict(...).
        Returns:
            attr_list: All S.M.A.R.T. attributes used in given disk. Here we
                       use intersection set of all disk days.
            diff_disk_days: A list struct comprises 5 dictionaries, each
                            dictionary contains differential attributes.
        Raises:
            Exceptions of wrong list/dict operations.
        """

        all_attrs = [set(disk_day.keys()) for disk_day in disk_days]
        attr_list = list(set.intersection(*all_attrs))
        prev_days = disk_days[:-1]
        curr_days = disk_days[1:]
        diff_disk_days = []
        # TODO: ensure that this ordering is correct
        for prev, cur in zip(prev_days, curr_days):
            diff_disk_days.append(
                {attr: (int(cur[attr]) - int(prev[attr])) for attr in attr_list}
            )

        return attr_list, diff_disk_days

    def __get_best_models(
        self, attr_list: AttrNamesT
    ) -> Optional[Dict[str, List[str]]]:
        """
        Find the best model from model list according to given attribute list.
        Args:
            attr_list: All S.M.A.R.T. attributes used in given disk.
        Returns:
            modelpath: The best model for the given attribute list.
            model_attrlist: 'Ordered' attribute list of the returned model.
                            Must be aware that SMART attributes is in order.
        Raises:
        """

        models = self.model_context.keys()

        scores = []
        for model_name in models:
            scores.append(
                sum(attr in attr_list for attr in self.model_context[model_name])
            )
        max_score = max(scores)

        # Skip if too few matched attributes.
        if max_score < 3:
            print("Too few matched attributes")
            return None

        best_models: Dict[str, List[str]] = {}
        best_model_indices = [
            idx for idx, score in enumerate(scores) if score > max_score - 2
        ]
        for model_idx in best_model_indices:
            model_name = list(models)[model_idx]
            model_path = os.path.join(self.model_dirpath, model_name)
            model_attrlist = self.model_context[model_name]
            best_models[model_path] = model_attrlist

        return best_models
        # return os.path.join(self.model_dirpath, model_name), model_attrlist

    @staticmethod
    def __get_ordered_attrs(
        disk_days: Sequence[DevSmartT], model_attrlist: List[str]
    ) -> List[List[float]]:
        """
        Return ordered attributes of given disk days.
        Args:
            disk_days: Unordered disk days.
            model_attrlist: Model's ordered attribute list.
        Returns:
            ordered_attrs: Ordered disk days.
        Raises: None
        """

        ordered_attrs = []

        for one_day in disk_days:
            one_day_attrs = []

            for attr in model_attrlist:
                if attr in one_day:
                    one_day_attrs.append(one_day[attr])
                else:
                    one_day_attrs.append(0)

            ordered_attrs.append(one_day_attrs)

        return ordered_attrs

    def predict(self, disk_days: Sequence[DevSmartT]) -> str:
        """
        Predict using given 6-days disk S.M.A.R.T. attributes.
        Args:
            disk_days: A list struct comprises 6 dictionaries. These
                       dictionaries store 'consecutive' days of disk SMART
                       attributes.
        Returns:
            A string indicates prediction result. One of following four strings
            will be returned according to disk failure status:
            (1) Good : Disk is health
            (2) Warning : Disk has some symptoms but may not fail immediately
            (3) Bad : Disk is in danger and data backup is highly recommended
            (4) Unknown : Not enough data for prediction.
        Raises:
            Pickle exceptions
        """

        all_pred = []

        proc_disk_days = self.__preprocess(disk_days)
        attr_list, diff_data = PSDiskHealthClassifier.__get_diff_attrs(proc_disk_days)
        modellist = self.__get_best_models(attr_list)
        if modellist is None:
            return "Unknown"

        for modelpath in modellist:
            model_attrlist = modellist[modelpath]
            ordered_data = PSDiskHealthClassifier.__get_ordered_attrs(
                diff_data, model_attrlist
            )

            try:
                with open(modelpath, "rb") as f_model:
                    clf = pickle.load(f_model)

            except UnicodeDecodeError:
                # Compatibility for python3
                with open(modelpath, "rb") as f_model:
                    clf = pickle.load(f_model, encoding="latin1")

            pred = clf.predict(ordered_data)

            all_pred.append(1 if any(pred) else 0)

        score = 2 ** sum(all_pred) - len(modellist)
        if score > 10:
            return "Bad"
        if score > 4:
            return "Warning"
        return "Good"
