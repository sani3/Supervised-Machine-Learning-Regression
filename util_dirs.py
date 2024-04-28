import os
import shutil
import urllib.error
import urllib.request


def create_project_dirs():
    """
    Create a directories 'resources', 'resources/data/' and 'resources/download/data' in the current working directory if not exist
    :return:
    """

    try:
        os.makedirs(os.path.join('resources'))
        print('Success: Directory created resources')
    except FileExistsError:
        print('Directory already exist: resources')

    try:
        os.makedirs(os.path.join('resources', 'data'))
        print('Success: Directory created resources/data')
    except FileExistsError:
        print('Directory already exist: resources/data')

    try:
        os.makedirs(os.path.join('resources', 'styles'))
        print('Success: Directory created resources/styles')
    except FileExistsError:
        print('Directory already exist: resources/styles')

    try:
        os.makedirs(os.path.join('resources', 'images'))
        print('Success: Directory created resources/images')
    except FileExistsError:
        print('Directory already exist: resources/images')

    try:
        os.makedirs(os.path.join('resources', 'download'))
        print('Success: Directory created resources/download')
    except FileExistsError:
        print('Directory already exist: resources/download')


if __name__ == 'main':
    # Setup relevant project directories
    create_project_dirs()
