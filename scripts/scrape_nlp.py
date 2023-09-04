import logging
import os
from argparse import ArgumentParser
from constants import PARENT_DIR, SAVE_DIR, USERS

from utils.scraper import GitHubScraper

logger = logging.getLogger(__name__)

def main():
    scraper = GitHubScraper(auth_username="ashtonomy", 
                            auth_token=os.environ["GH_TOKEN"])
    
    data = scraper(users=USERS, hidden_files=False, file_endings="nlp",
                   save_dir=SAVE_DIR)

    print(f"{len(data)} total samples in dataset")
    if isinstance(data, dict):
        total_lines = 0
        for k, v in data.items():
            total_lines += len(v.split("\n"))
        print(f"{total_lines} total lines in dataset")
    
    elif isinstance(data, list):
        total_lines = 0
        for v in data:
            total_lines += len(v.split("\n"))
        print(f"{total_lines} total lines in dataset")

if __name__ == '__main__':
    main()
