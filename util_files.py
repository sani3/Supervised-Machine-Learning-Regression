import os
import shutil
import urllib

import requests


def download_file(url, root=os.path.join('.', 'resources', 'download')):
    """
    Download file to 'resources/download' from url
    :param root:
    :param url:
    :return:
    """
    filename = url.split('/')[-1]
    filename_path = os.path.join(root, filename)
    if not os.path.exists(os.path.join(root, filename)):
        with requests.get(url, stream=True) as req:
            req.raise_for_status()
            with open(filename_path, 'wb') as conn:
                for chunk in req.iter_content(chunk_size=8192):
                    conn.write(chunk)
        print('Download: Successful!')
        return filename_path
    else:
        print('Download: File already exists!')
        return filename_path


def extract(filename_path, root=os.path.join('.', 'resources', 'download')):
    """
    Extract from archive
    :param filename_path:
    :param root:
    :return:
    """
    if os.path.exists(filename_path):
        try:
            shutil.unpack_archive(filename_path, root)
            print('Extraction: Successful!')
        except shutil.Error:
            print('Extraction failed: Something went wrong')
    else:
        print('Does not exist: ', filename_path)


# def download_unpack(url):
#     """
#     Download file to 'resources/download' from url
#     :param url:
#     :return:
#     """
#     file_name = url.split("/")[-1]
#     download_path = os.path.join('resources', 'download')
#     download_file = os.path.join(download_path, file_name)
#     if not os.path.exists(download_file):
#         try:
#             urllib.request.urlretrieve(url, download_file)
#             print("Success: File downloaded " + download_file)
#         except urllib.error.URLError:
#             print("Something went wrong: Could download file")
#     else:
#         print('File already exist: ' + download_file)
#
#     try:
#         shutil.unpack_archive(download_file, download_path)
#         print("Success: File unpacked")
#     except shutil.Error:
#         print("Something went wrong: Could unpack file")


if __name__ == 'main':
    # Download and extract archived dataset
    url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz'
    extract(download_file(url))
    # download_unpack(url)
