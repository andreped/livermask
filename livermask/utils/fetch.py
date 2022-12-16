import os
import requests
from datetime import datetime
from email.utils import parsedate_to_datetime, formatdate


# taken from: https://lenon.dev/blog/downloading-and-caching-large-files-using-python/
def download(url, destination_file):
    headers = {}

    if os.path.exists(destination_file):
        mtime = os.path.getmtime(destination_file)
        headers["if-modified-since"] = formatdate(mtime, usegmt=True)

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    if response.status_code == requests.codes.not_modified:
        return

    if response.status_code == requests.codes.ok:
        with open(destination_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1048576):
                f.write(chunk)
        
        last_modified = response.headers.get("last-modified")
        if last_modified:
            new_mtime = parsedate_to_datetime(last_modified).timestamp()
            os.utime(destination_file, times=(datetime.now().timestamp(), new_mtime))
