# coding=utf8

import json
import copy
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                if isinstance(value,dict):
                    setattr(self, key, Config(**value))
                else:
                    setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def from_dict(cls, config_dict:Dict[str,Any], **kwargs) -> "Config":
        config = cls(**config_dict)
        return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "Config":
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls,json_path: str):
        with open(json_path, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return json.loads(text)