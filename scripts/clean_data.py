import pickle
import json
import os
import re
import typing
from typing import List, Union, Dict
from argparse import ArgumentParser
import logging
from constants import SAVE_DIR
from datasets import Dataset
from dateutil import parser

logger = logging.getLogger(__name__)

# Proper nouns to remove frmo comments (lowercase)
REMOVE_NAMES = [
    'david', 
    'amnon', 
    'ashton', 
    'elvis' # Stop leaving hurtful comments in my dataset, elvis.
]

def remove_header(sample: str) -> str:
    lines = sample.split('\n')
    header_end_idx = 0
    header_seps = 0
    while lines[header_end_idx].startswith('#') or lines[header_end_idx] == '':
        if header_end_idx >= len(lines):
            return ""
        if lines[header_end_idx].startswith('####'):
            header_seps += 1
        header_end_idx += 1
        if header_seps >= 2:
            break
    
    return "\n".join(lines[header_end_idx:])

def resolve_line_endings(sample: str) -> str:
    """
    Convert carriage returns to newlines, handling line 
    ending literals within rules. Has not been tested. 
    There are better solutions for this, like zip/unzip on linux.
    """
    if "\r" not in sample:
        return sample
    
    clean_sample = sample.replace("\r\r\n", "\n")
    clean_sample = clean_sample.replace("\r\n", "\n") 
    return clean_sample

def de_identify(sample: str) -> str:
    """
    Remove dates, times, names, emails, and websites from comments.
    Not rigorous.
    """
    lines = sample.split("\n")
    clean_lines = []
    for l in lines:
        if l.lstrip().startswith("#"):
            # find datetime matches
            dt_match = re.search(r'\d{1,4}[/-:]\d{1,2}[/-:]?\d{0,4}?', l.lower())
            if dt_match:
                l = l[:dt_match.start()].rstrip()
                if l.endswith("-"):
                    l = l[:-1].rstrip()

            # Find name matches    
            name_match = re.search('|'.join(REMOVE_NAMES), l.lower())
            if name_match:
                l = l[:name_match.start()].rstrip()
                if l.endswith("-"):
                    l = l[:-1].rstrip()
            
            # Find a url match. should also get email addresses.
            # Ignores case with multiple urls. 
            url_match = re.search(r'https?://\S*|www\.\S+\.\S*|\S+\.com\S+|\S+\.net\S+|\S+\.org\S+|\S+\.edu\S+|\S+\.uk\S+', l.lower())
            if url_match:
                l = l[:url_match.start()] + l[url_match.end():]
                        
            # Only keep cleaned line if it contains some info
            # E.g. After cleaning, we should keep the line < # Iterate over nodes - ashton 1/1/1901 >
            if name_match or dt_match or url_match:
                if len(l.strip()) > 10:
                    clean_lines.append(l)
            else:
                clean_lines.append(l)

        else:
            # Remove line ending date stamps like # 02/12/02 AM.
            date_stamp_match = re.search(r'#\s*\d{2}/\d{2}/\d{2} [AP]M\.', l)
            if date_stamp_match:
                l = l[:date_stamp_match.start()].rstrip()

            clean_lines.append(l)
        
    return "\n".join(clean_lines)

def clean_data(data: List[str]) -> Dict[int, str]:
    clean_data = []
    for sample in data:
        clean_sample = resolve_line_endings(sample)
        clean_sample = remove_header(clean_sample)
        clean_sample = de_identify(clean_sample)
        if clean_sample != "":
            clean_data.append(clean_sample)

    clean_data = {i: sample for i, sample in enumerate(clean_data)}
    return clean_data

def read_data(pickled_data_path: str) -> List[str]:
    with open(pickled_data_path, "rb") as f:
        return pickle.load(f)

def save_clean_data(data: Dict[int, str], save_path: str, force: bool=False) -> None:
    if not force and os.path.exists(save_path):
        raise FileExistsError("save_path exists. Set force = True to overwrite.")

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

def clean(
    raw_data: List[str]=None,
    pickled_data_path: str=None,
    save_path: str=None,
    force: bool=False
) -> Dict[int, str]:
    """
    Read pickled data, remove headers from samples, save as json, 
    and return clean data.

    Args
        pickled_data_path: path to pickled data file
        save_path: name of save file
        force: whether to overwrite save_path, if it exists
    
    Return
        dict of sample idx -> sample string
    """
    if raw_data is None and pickled_data_path is not None:
        data = read_data(pickled_data_path)
    else:
        data = raw_data
    data = clean_data(data)
    if save_path is not None:
        save_clean_data(data, save_path, force)
    return data

def save_hf_dataset(data_dict: Dict[int, str], save_path: str) -> Dataset:
    dataset = Dataset.from_dict({"text": list(data_dict.values())})
    if save_path.endswith(".json"):
        save_path = save_path.replace(".json", "")
    dataset.save_to_disk(save_path)
    return dataset

def parse_args():
    arg_parser = ArgumentParser(description = "Clean raw scraped data.")
    arg_parser.add_argument(
        "--pickled_data_path",
        type=str,
        default=None,
        help="Path to pickled data file to load"
    )
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to write cleaned json data"
    )
    arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to overwrite save path."
    )
    return arg_parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    _ = clean(pickled_data_path=args.pickled_data_path, 
              save_path=args.save_path, force=args.force)
