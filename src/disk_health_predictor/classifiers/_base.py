"""Abstract base class for all classifiers. Defines the API for classifiers."""
from abc import abstractmethod
from typing import Dict


class DiskHealthClassifier:
    """
    Base class for classifiers (good/bad/warning)
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def initialize(self, model_dir: str) -> None:
        raise NotImplementedError()

    @classmethod
    def extract_ata_smart_attrs(cls, health_data: Dict) -> Dict:
        """
        Extract data from smartctl attribute table in from smartctl json.
        """
        # dummy vars to be used later. avoid overhead of created multiple empty objects
        emptydict = {}
        emptylist = []

        # init empty dict for each date
        flattened_smart_data = dict((d, dict()) for d in health_data.keys())

        # parse data for each day and save it in the flattened data dict
        for date, date_smart_json in health_data.items():
            for attr in date_smart_json.get("ata_smart_attributes", emptydict).get(
                "table", emptylist
            ):
                # smart stat number
                attr_id = attr["id"]

                # its better to extract raw value from string key instead of value key
                # in some cases, raw value looks like this -
                # {'value': 471138330, 'string': '26 (Min/Max 21/28)'}
                # here, the correct value should be 26 and not 471138330
                raw_str = attr.get("raw", emptydict).get("string", None)
                if raw_str is not None:
                    raw_str = raw_str.split(" ", maxsplit=1)[0]

                # try to parse value from string
                if raw_str.isdigit():
                    raw = int(raw_str)
                else:
                    # if not possible, use the number in "value" key
                    # case in point - "string": "7019h+59m+24.383s"
                    # TODO: also throw warning
                    raw = attr.get("raw", emptydict).get("value")

                # save raw and normalized values
                flattened_smart_data[date][f"smart_{attr_id}_raw"] = raw
                flattened_smart_data[date][f"smart_{attr_id}_normalized"] = attr.get(
                    "value", None
                )

            # try to add power on hours manually
            # in case it wasnt extracted from smart attribute table
            if flattened_smart_data[date].get("smart_9_raw") is None:
                flattened_smart_data[date]["smart_9_raw"] = date_smart_json.get(
                    "power_on_time", emptydict
                ).get("hours")

            # add device capacity
            user_capacity = date_smart_json.get("user_capacity", emptydict).get("bytes")
            if isinstance(user_capacity, dict):
                user_capacity = user_capacity["n"]
            flattened_smart_data[date]["user_capacity"] = user_capacity

        return flattened_smart_data

    @abstractmethod
    def predict(self, dataset) -> str:
        raise NotImplementedError()
