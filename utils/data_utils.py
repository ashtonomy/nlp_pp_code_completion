"""
Misc utilities for handling text data. Includes:
 - scrape_dir(dir_path, file_endings, filter_duplicates, save_path)
"""
import os
import json
import pickle
import typing
from typing import (
    Union, 
    Optional,
    List,
    Dict
)
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def scrape_dir(
    dir_path: str, 
    file_endings: Optional[Union[str, List[str]]]=None,
    filter_duplicates: bool=True,
    save_path: Optional[str]=None
) -> Dict[str, str]:
    """
    Retrieve files from directory recursively
    Assumes file contents in dir are small enough to fit in memory

    Args
        dir_path: Directory to scrape
        file_endings: target filename suffixes (e.g.: ["csv", ".txt"])
        filter_duplicates: whether to keep files with duplicated content
        save_path: If not None, file path to save data to. Must not exist.
    Returns 
        Dictionary of form {"file_path": "raw file text",...}
    """
    
    # Get all file paths, filtered by file endings (or none)
    file_paths = []
    if file_endings is not None:
        for ending in file_endings:
            file_paths += list(Path(dir_path).rglob("*.nlp"))
    else:
        file_paths += [i for i in Path(dir_path).rglob("*") if i.is_file()]

    # To ensure unique contents, file text is key
    data = {}

    for path in tqdm(file_paths, desc="Reading data..."):
        path = str(path)
        try:
            with open(path, "r") as f:
                text = f.read()

                # If filter_duplicates, keep first file path read
                if not filter_duplicates:
                    data[path] = text
                elif text not in data:
                    data[text] = path
                else:
                    logger.debug(f"Filtering duplicate file: {path}")
    
        except Exception as e:
            logger.warn(f"Unable to read file at {path}:")
            logger.warn(e)

    # invert dict to get file path to file text
    if filter_duplicates:
        data = {v:k for k, v in data.items()}

    if save_path is not None:
        if save_path.endswith(".pkl"):
            with open(save_path, "wb") as f:
                pickle.dump(data, file)
        elif save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            logger.warn("Save file type not supported.")
            logger.warn("Must be either '.json' or '.pkl'")

    return data

def merge_datasets(*args):
    """
    Merge text, removing duplicates. Order is not preserved.

    Args:
        Each arg should be either text list dict with text values

    Returns:
        Dict of form {"text": [0,...,n_unique_samples]}
    """

    merged_data = set()
    for arg in args:
        if isinstance(arg, list):
            merged_data.update(arg)
        elif isinstance(arg, dict):
            merged_data.update(list(arg.values()))

    merged_data = {"text": list(merged_data)}
    return merged_data

