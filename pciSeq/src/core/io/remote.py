"""Fetching the example datasets over http."""

import os
import shutil
import tempfile
import logging
from urllib.parse import urlparse
from urllib.request import urlopen

from tqdm import tqdm

logger = logging.getLogger(__name__)



def download_url_to_file(url: str, dst: str, progress: bool = True) -> None:
    """Download object at the given URL to a local path.

    Args:
        url: URL of the object to download
        dst: Full path where object will be saved
        progress: Whether to display a progress bar

    Notes:
        Thanks to torch, slightly modified
    """
    file_size = None
    u = urlopen(url)
    meta = u.info()

    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")

    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # Save to temp file first then move
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)



def load_from_url(url: str) -> str:
    """Download file from URL if not already present locally.

    Args:
        url: URL to download from

    Returns:
        str: Local filename
    """
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if not os.path.exists(filename):
        logger.info('Downloading: "%s" to %s', url, filename)
        download_url_to_file(url, filename)
    return filename
