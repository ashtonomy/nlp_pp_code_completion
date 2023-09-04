import requests
from urllib.error import HTTPError
import pandas as pandas
import math
import re
import os
import typing 
from typing import Union, List, Tuple, Dict
from tqdm import tqdm
import chardet
import logging
import warnings
import pickle
import base64


logger = logging.getLogger(__name__)

class GitHubScraper:
    def __init__(
        self,
        base_url: str="https://api.github.com/",
        auth_username: str=None,
        auth_token: str=None,
        use_tqdm: bool=True
    ):
        """Initialize GitHubScraper
 
        Args:
            base_url: Github base url for scraping, defaults to github api url 
            auth_username: username for authentication for github api requests
            auth_token: token for authentication for github api requests
            use_tqdm: whether to display tqdm progress bar (only visible in __call__)
        """
        self.base_url = base_url
        if auth_username is not None or auth_token is not None:
            self.auth = (auth_username, auth_token)
        else:
            self.auth = None
        self.use_tqdm = use_tqdm

    def __call__(
        self,
        users: Union[str, List[str]],
        get_forks: bool=False,
        branch_or_commit_hash: str='main',
        hidden_files: bool=True,
        file_endings: Union[str, List[str]]=None,
        remove_duplicates: bool=True,
        save_dir: str=None,
        file_name: str=None
    ) -> List[str]:
        """Alias for self.scrape
        """
        
        return self.scrape(
            users = users,
            get_forks = get_forks,
            branch_or_commit_hash = branch_or_commit_hash,
            hidden_files = hidden_files,
            file_endings = file_endings,
            remove_duplicates = remove_duplicates,
            save_dir = save_dir,
            file_name = file_name
        )


    def scrape(
        self,
        users: Union[str, List[str]],
        get_forks: bool=False,
        branch_or_commit_hash: str='main',
        hidden_files: bool=True,
        file_endings: Union[str, List[str]]=None,
        remove_duplicates: bool=True,
        save_dir: str=None,
        file_name: str=None
    ) -> List[str]:
        """
        Get all files from user or list of users
        """

        if isinstance(users, str):
            users = [users]

        if remove_duplicates:
            all_data = set()
        else:
            all_data = []

        for user in users:
            repos = self.get_repos(user=user, get_forks=get_forks, 
                                   urls_only=False)

            repo_urls = []
            for repo in repos:
                repo_url = self.resolve_path_parameters(repo["trees_url"], 
                                                        "sha", 
                                                        repo["default_branch"])
                repo_urls.append(repo_url)
            
            if self.use_tqdm:
                pbar = tqdm(repo_urls, desc=f'Retrieving files for user {user}')
            else:
                logger.info(f'Retrieving files for user {user}')
                pbar = repo_urls
            
            all_file_urls = []
            for repo_url in pbar:
                file_urls = self.get_repo_files(repo_url=repo_url, 
                                           branch_or_commit_hash=None, # Already added.
                                           hidden_files=hidden_files,
                                           file_endings=file_endings,
                                           urls_only=True)

                all_file_urls += file_urls
            
            logger.info(f'Retrieving file data for user {user}')

            data = self.get_data(all_file_urls)
            # data = list(data)

            if remove_duplicates:
                all_data.update(data)
            else:
                all_data += data

        all_data = list(all_data) if remove_duplicates else all_data

        if save_dir is not None:
            if file_name is None:
                file_name = "_".join(users)[:100] + "_"
                file_name += "_".join(file_endings) + "_github_data.pkl"

            self.save_data(data=all_data, save_dir=save_dir, file_name=file_name)
        
        return all_data

    @staticmethod
    def resolve_path_parameters(
        url: Union[str, List[str]],
        param_name: str,
        value: str
    ) -> Union[str, List[str]]:
        """
        Replace path parameters in endpoint with value str.

        Args:
            url: string url or list of str urls
            param_name: param str to replace
            value: Value to replace with name with
        Returns
            list of str urls or str url with param name replaced by value
        """
        param_name = param_name.replace("{", "").replace("}", "")
        param_name.replace("/", "")

        if isinstance(url, str):
            resolved_url = url.replace("{"+param_name+"}", value)
            resolved_url = resolved_url.replace("{/"+param_name+"}", "/"+value) 
            return resolved_url
        
        resolved_urls = []
        for endpoint in url:
            resolved_url = endpoint.replace("{"+param_name+"}", value)
            resolved_url = resolved_url.replace("{/"+param_name+"}", "/"+value)
            resolved_urls.append(resolved_url)
        return resolved_urls

    @staticmethod
    def join_url(
        base_url: str=None,
        *args
    ) -> str:
        """Contatenate string paramers into a url.
        E.g. join_url("https://www.github.com/", "/user", "repo/")
                -> "https://www.github.com/user/repo/"
        """
        if '://' in base_url:
            if not base_url.endswith('://'):
                prefix, postfix = base_url.split('://', 1)
                url = prefix + '://' + re.sub('/+', '/', postfix)
                url = url.rstrip('/')
            else:
                # Assume that base_url is of form "https://". No validation is done here.
                url = base_url
        else:
            url = re.sub('/+', '/', base_url)
            url = url.rstrip('/')

        for idx, loc in enumerate(args):
            if idx != len(args)-1:
                url += '/' + re.sub('/+', '/', loc).strip('/')
            else:
                url += '/' + re.sub('/+', '/', loc).lstrip('/')

        return url
    
    def get_user(
        self, 
        user: str
    ) -> requests.Request:
        """
        Get user request
        """   
        url = self.join_url(self.base_url, 'users', user)
        r = requests.get(url, auth=self.auth)
        if r.status_code == 200: 
            logger.debug(f"{r.status_code} response for GET user request from {url}")
        else:
            logger.warning(f"{r.status_code} response from {url}")
        return r

    def get_repos(
        self,
        user: str,
        get_private_repos: bool=False,
        get_forks: bool=False,
        urls_only: bool=False,
        trees_urls: bool=False
    ) -> Union[List[dict], List[str]]:
        """GET request from:
        https://api.github.com/users/{user}/repos

        Add additional function to handle multiple users.
        """
        url = self.join_url(self.base_url, 'users', user, 'repos')
        r = requests.get(url, auth=self.auth)
        status = r.status_code
        if status == 200: 
            logger.debug(f"{status} response for GET public repos request from {url}")
        else:
            logger.warning(f"{status} response from {url}: unable to get repos")
            return []
        
        all_repos = r.json()
        logger.info(f"Retrieving data for {len(all_repos)} repos from user {user}")

        repos = []
        url_type = 'url' if not trees_urls else 'trees_url'
        for repo in all_repos:
            fork = repo["fork"]
            private = repo["private"]
            size = repo["size"]

            if not private and size > 0:
                if get_forks:
                    repos.append(repo if not urls_only else repo[url_type])
                elif not fork:
                    repos.append(repo if not urls_only else repo[url_type])

        return repos

    def get_latest_commit(
        self,
        url: str,
        branch: str='main'
    ) -> str:
        warnings.warn("get_latest_commit is not implemented.")
        return None
    #     url = self.join_url(self.base_url, 'users', user, 'repos')
    #     response = requests.get(url)
    #     if response.status_code == 200: 
    #         logger.debug(f"{response.status_code} response for GET public repos request from {url}")
    #     else:
    #         logger.warning(f"{response.status_code} response from branch {url}: unable to get repositories")
    #         return []
    #     pass

    @staticmethod
    def format_file_endings(
        file_endings: Union[str, List[str]]
    ) -> List[str]:
        """
        Ensures 1) that file_endings are a list and 
                2) that file endings don't contain a period 
        """
        if isinstance(file_endings, str):
            file_endings = [file_endings]
        new_endings = []
        for f in file_endings:
            if '.' not in f:
                new_endings.append(f)
            else:
                fixed_file_ending = f.rsplit(".")[-1]
                if fixed_file_ending != "":
                    new_endings.append(fixed_file_ending)
        return new_endings

    def get_repo_files(
        self,
        repo_url: str,
        branch_or_commit_hash: str=None,
        hidden_files: bool=True,
        file_endings: Union[str, List[str]]=None,
        urls_only: bool=True
    ) -> Union[List[dict], List[str]]:
        """
        Get files from repo at branch main.

        Args
            repo_url: Github API endpoint for repo. Can also be tree url.
            branch: Branch or commit hash
            hidden_files: Whether to retrieve hidden files.
            file_endings: restrict to a certain file type(s)
            urls_only: Whether to return only urls of files
        """
        if "git/trees" in repo_url:
            url = repo_url
        else:
            url = self.join_url(repo_url, "git", "trees")
        if branch_or_commit_hash is not None:
            url = self.join_url(url, branch_or_commit_hash)
        if "?recursive" not in url:
            url += "?recursive=1"

        r = requests.get(url, auth=self.auth)
        status = r.status_code
        if status == 200: 
            logger.debug(f"{status} response for GET repo contents from {url}")
        else:
            logger.warning(f"{status} response from {url}: unable to get contents")
            return []

        contents = r.json()['tree']
        if file_endings is not None:
            file_endings = self.format_file_endings(file_endings)

        files = []
        for item in contents:
            if item["type"] == 'blob':
                if file_endings is not None:
                    suffix = item["path"].rsplit(".")[-1]
                    if suffix in file_endings:
                        files.append(item if not urls_only else item["url"])
                else:
                    files.append(item if not urls_only else item["url"])
        return files

    def get_data(
        self,
        file_urls: Union[str, List[str]],
        raise_content_errors: bool=False,
        raise_decode_errors: bool=False
    ) -> Dict[str, str]:
        """
        Given list of files, Returns text data from files. 

        Raises
            KeyError if response contains no "content"
        """
        if isinstance(file_urls, str):
            file_urls = [file_urls]

        if self.use_tqdm:
            pbar = tqdm(file_urls, desc=f'Retrieving file data')
        else:
            logger.info(f'Retrieving file data')
            pbar = file_urls

        texts = []
        for file_url in pbar:
            data = requests.get(file_url, auth=self.auth)
            try:
                content = data.json()["content"]
            except KeyError as e:
                logger.warning(f"No content detected for file {file_url}")
                if raise_content_errors:
                    raise e
                else:
                    continue
                
            try:
                decoded_text = base64.b64decode(content)
                texts.append(decoded_text.decode())
                # texts[file_url] = decoded_text.decode()
            except Exception as e:
                warnings.warn(f"Unable to decode content for file at: {file_url}")
                if raise_decode_errors:
                    raise e 

        logger.info(f'Decoded data for {len(texts)}/{len(file_urls)} files')
            
        return texts

    def save_data(
        self,
        data: Union[str, list[str], dict],
        save_dir: str=None,
        file_name: str=None
    ) -> None:
        """
        Pickle scraped text data.
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, file_name)
        logger.info(f"Saving scraped data to {save_path}.")
        with open(save_path, "wb") as f:
            pickle.dump(data, f)