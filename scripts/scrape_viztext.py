"""
Scrape Visual Text help files to generate prompt dataset.
"""
import os
import requests
from bs4 import BeautifulSoup, SoupStrainer
import typing
from typing import Union, Dict, List, Optional
from urllib.parse import urljoin
import logging
import json
from tqdm import tqdm
from pprint import pprint

logger = logging.getLogger(__name__)

def get_links(url: str, link_ending: str="htm") -> Dict[str,str]:
    avoid_start_chars = ["$", "@", "_"] # These contain no examples and but may still slip through
    response = requests.get(url)

    links = []
    for link in BeautifulSoup(response.content, parse_only=SoupStrainer('a')):
        if link.has_attr('href'):
            if link['href'].endswith(link_ending) and link['href'].islower() and link['href'][0] not in avoid_start_chars:
                # Create absolute link, assumes relative links
                abs_link = urljoin(url, link['href'])
                links.append(abs_link)
    return links

def get_content(url: str) -> Optional[List[str]]:
    """
    Get title, purpose, and example from url page.
    """
    content = {}

    response = requests.get(url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    target = soup.find('h2',string=lambda text: text.strip().lower() == "example")
    if target is None:
        return None
    
    example_body = []
    for sib in target.find_next_siblings():
        if sib.name not in [None, 'p']:
            break
        else:
            example_text = sib.text

            # In these cases an image or output file snippet follows, so we stop.
            if example_text == "\xa0":
                break
            if not example_text.strip().startswith("#"):
                if ";" not in example_text and " output.txt" in example_text.lower():
                    break
                if "this prints " in example_text.lower() and ";" not in example_text:
                    break
                if "kb editor" in example_text.lower():
                    break
            
            example_body.append(example_text)
        
    # Example with single line text caption and no code.
    if len(example_body) == 1:
        if not example_body[0].lstrip().startswith("#") and ";" not in example_body[0] and '"' not in example_body[0] and "=" not in example_body[0] and "<-" not in example_body[0]:
            return None
    # Empty block example
    elif len(example_body) == 2:
        if len(example_body[0]) < 15 and len(example_body[1]) < 15:
            if "@" in example_body[0] and "@@" in example_body[1]:
                return None
    
    content["example"] = example_body
    title = soup.find('title')
    content["title"] = title.text if title is not None else None

    purpose_target = soup.find('h2',string=lambda text: text.strip().lower() == "purpose")
    purpose = []
    if purpose_target is not None:
        for sib in purpose_target.find_next_siblings():
            if sib.name not in [None, 'p']:
                break
            else:
                purpose.append(sib.text)

        content["purpose"] = purpose
    else:
        content["purpose"] = None

    return content

def scrape(index_page_url: str):
    help_pages = get_links(index_page_url)
    if len(help_pages) == 0:
        raise ValueError("No links found on index page.")
    
    content = {}
    for page in tqdm(help_pages, desc="Scraping web pages"):
        page_content = get_content(page)
        if page_content is None:
            logger.warn(f"Unable to locate example on page <{page}>")
        else:
            content[page] = page_content

    return content

def clean_examples(example_dict: Dict[str, Dict[str,str]]) -> Dict[str, Dict[str,str]]:
    """
    Clean up examples
    """
    clean_dict = {}
    for k, v in example_dict.items():
        raw_text = v["example"]
        clean_text = []
        for i, line in enumerate(raw_text):
            # Probably a text line. definitely not perfect.
            if i == 0 and ";" not in line and "@@" not in line and " = " not in line and not line.lstrip().startswith("#"):
                line = line.replace("\n", " ").replace("  ", " ")

            new_line = line.replace("If(", "if(").replace("While(", "while(")
            new_line = new_line.replace("\n<<", " <<").replace("  <<", " <<")

            # Naive approach to indent bracketed blocks. Nested brackets are rare, if they exist at all
            if "{" in new_line:
                pre_brack, post_brack = new_line.split("{", 1)
                if "}" in post_brack:
                    interior, exterior = post_brack.split("}", 1)
                    interior = interior.replace("\n", "\n\t").replace('\\n\t', "\\n") # <- Handle escaped newlines
                    if interior.endswith("\t"):
                        interior = interior[:-1]
                    new_line = pre_brack + "{" + interior + "}" + exterior

            clean_text.append(new_line)
                
        v["example"] = clean_text
        clean_dict[k] = v
    return clean_dict

import copy
import sys

def get_prompts(example_dict: Dict[str, Dict[str,str]]) -> Dict[str, Dict[str,str]]:
    """
    Attempt to create prompts for examples
    """
    formatted_dict = {}
    for k, v in example_dict.items():
        example = v["example"]
        prompt = []
        purpose = v["purpose"] if "purpose" in v else None
        if purpose is not None and "".join(v["purpose"]).lower().startswith("obsolete"):
            continue

        if example == [] or example == '':
            continue
        
        first_line = example[0]
        if first_line.strip().startswith("#"):
            sublines = first_line.split("\n")
            for l in sublines:
                if l.startswith("#"):
                    prompt.append(l)
                else:
                    break
            for x in prompt:
                sublines.remove(x)
            example = sublines
        elif ";" not in first_line and "<-" not in first_line and " = " not in first_line and "@" not in first_line:
            prompt.append(first_line)
            example = example[1:]
        else:
            if "purpose" in v:
                prompt.append(v["purpose"])
            
        if prompt != []:
            prompt = [" ".join(p) if isinstance(p, list) else p for p in prompt]
            prompt = ". ".join(prompt).replace("#", "").replace("  ", " ").strip()
            
            # E.g. In this example, we define a function to...
            if "," in prompt:
                pre, post = prompt.split(",",1)
                if len(pre) < 20:
                    prompt = post.strip()
            if prompt.lower().startswith("we "):
                prompt = prompt[3:]
            
            prompt = "# " + prompt
            v["prompt"] = prompt
        else:
            name = v["title"]
            logger.warn(f"Unable to create prompt for {name}. Consider adding manually.")
        
        v["example"] = "\n".join(example)

        formatted_dict[k] = v
    
    return formatted_dict


def scrape_and_process(index_page_url: str, save_path: str=None) -> Dict[str, str]:
    """
    Scrape help pages, clean resuling data and generate prompts, if possible.
    """
    content = scrape(index_page_url)
    clean_content = clean_examples(content)
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(clean_content, f, indent=4)
    content_w_prompts = get_prompts(clean_content)
    return content_w_prompts

def flatten(data_dict: Dict[str, Dict[str,str]], key_col_name: str="url", null_val=None) -> Dict[str, List[str]]:
    """
    Convert nested dictionary to dict of form {"col1": [...], "col2": [...],...}
    Args
        data_dict: nested dataset to flatten
        null_val: value to use for nonexistent column val
    """
    flattened = {}
    first_iter = True
    for k, v in data_dict.items():
        if first_iter:
            flattened[key_col_name] = [k]
        else:
            flattened[key_col_name].append(k)
        
        col_length = len(flattened[key_col_name])
        for field, val in v.items():
            if isinstance(val, list):
                clean_val = "\n".join(val)
            else:
                clean_val = val
            if first_iter:
                flattened[field] = [clean_val]
            else:
                if field not in flattened:
                    col = [null_val] * col_length
                    col[-1] = clean_val
                    flattened[field] = col
                else:
                    flattened[field].append(clean_val)
        
        if not first_iter:
            for k, v in flattened.items():
                if len(v) != col_length:
                    flattened[k].append(null_val)

        first_iter = False

        
    return flattened                

def run(
    index_page_url: str, 
    save_path: Optional[str]=None,
    flatten_dataset: bool=True
) -> Union[Dict[str, List[str]], Dict[str, Dict[str,str]]]:
    """
    Scrape, process, save, and return
    """
    data = scrape_and_process(index_page_url, save_path=save_path.replace(".json", "_clean.json"))
    data = flatten(data) if flatten_dataset else data
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)
    return data

