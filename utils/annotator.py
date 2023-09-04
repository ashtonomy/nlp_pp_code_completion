"""
Annotator

Annotator for .nlp files. Facilitates the addition of comments 
to nlp pass files to prepare for code generation clm training.
"""

import pandas as pandas
import math
import re
import os
import typing 
from typing import Union, List, Tuple, Dict
from tqdm import tqdm
import logging
import warnings
import pickle
from datasets import Dataset

logger = logging.getLogger(__name__)


class Annotator:
    def __init__(self, dataset: Union[List[str], dict]):
        """
        Initialize annotator with nlp++ dataset.
        """
        self.dataset = dataset
        self.current_index = 0
        self.finished = False
        warnings.warn('utils.annotator.Annotator is deprecated.', DeprecationWarning, stacklevel=2)

class Dataset:
    def __init__(self, dataset: Union[List[str], dict], text_column_name: str="text"):
        if isinstance(dataset, list):
            self._dataset = {text_column_name: dataset}
        else:
            self._dataset = dataset
        
        self.text_column_name = text_column_name
        self.column_names = [text_column_name]

        warnings.warn('utils.annotator.Dataset is deprecated.', DeprecationWarning, stacklevel=2)
        
    def __getitem__(self, key: Union[slice, int]):
        if isinstance(key, slice):
            result = {}
            for k, v in self._dataset:
                result[k] = v[key]
        elif isinstance(key, int):
            result = {}
            for k, v in self._dataset:
                result[k] = v[key]
        else:
            raise ValueError
        
        return result
    
    def __setitem__(self, key: int, value: Union[str, dict]):
        if isinstance(value, str):
            if len(self._dataset) > 1:
                logger.warn(f"Columns in value (1) must match dataset ({len(self._dataset)}).")
                raise ValueError
            self._dataset[self.text_column_name][key]

        elif isinstance(value, dict):
            if set(value.keys()) != set(self._dataset):
                logger.warn(f"Columns in value (1) must match dataset ({len(self._dataset)}).")
                raise ValueError
            
        else:
            raise ValueError("Value must be type str or dict.")

    def __len__(self):
        return len(self._dataset[self.text_column_name])

    def _set_column_names(self):
        """
        Update column_names.
        """
        self.column_names = list(self._dataset.keys())

    def rename_columns(self, column_mapping: Dict[str,str]):
        """
        Rename column(s) using mapping. 
        """
        for k, v in column_mapping.items():
            if k not in self._dataset:
                raise KeyError(f"Column '{k}' not in dataset.")
            if v in self.dataset:
                raise ValueError(f"Column '{v}' already in dataset.")
            self._dataset[v] = self._dataset[k]
            del self._dataset[k]

        self._set_column_names()
    
    def to_hf_dataset(self):
        """
        Convert dataset to datasets.Dataset
        """
        return Dataset.from_dict(self._dataset)

    def save_dataset(self, save_path: str):
        """
        pickle dataset
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load_dataset(cls, dataset_path: str) -> Dataset:
        """
        Load saved dataset from file.
        """
        with open(save_path, "rb") as f:
            dataset = pickle.load(f)
            return cls(dataset["_dataset"], dataset["text_column_name"])

