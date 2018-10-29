import os
import json
from time import sleep
from urllib import request as req

def download_data(max_download=5, pause=1, verbose=True):
    '''
    Download the data specified in /data/index.json into the data directory. This method will skip downloads if it finds that the work exists in the data directory already. The names of the files has the following structure (spaces are replaced with "_"),
        key_catalogue_m.mid

    Args:
        max_download (int): The maximum number of works to download from the index.json,
        pause (int): How long in seconds to pause between each download to prevent overloading remote server.
    Returns:
        int: Number of works downloaded.
    '''
    downloaded = 0
    index = __load_index__()
    works = index['works']
    for work in works:
        for i, mvmt in enumerate(work['movements']):
            if downloaded >= max_download:
                if verbose:
                    total_indexed, total_downloads = __file_stats__()
                    print("Downloaded {} for this run, {}/{} downloaded."
                            .format(downloaded, total_downloads, total_indexed))
                return downloaded

            name = __get_file_name__(work['key'], work['catalogue'], i + 1)
            downloaded_files = __get_all_downloaded_file_names__()
            if name in downloaded_files or bool(mvmt.get('downloaded', False)):
                if verbose:
                    print("Already downloaded {}, skipped.".format(name))
                continue

            url = mvmt['url']
            success, _ = __download_file__(url, name, "data/", verbose)
            if not success: return print("Terminating download.")
            downloaded += 1
            sleep(pause)

def list_downloaded_data():
    '''
    Returns:
        list: A list of strings containing the names of the downloade files.
    '''
    return __get_all_downloaded_file_names__()

def __file_stats__():
    '''
    Returns:
        (int, int): The number of files recorded in index.json, and the number of files that are downloaded locally.
    '''
    indexed_files = __get_all_indexed_file_names__()
    downloaded_files = __get_all_downloaded_file_names__()
    return len(indexed_files), len(downloaded_files)

def __get_all_downloaded_file_names__():
    '''
    Returns:
        list: A list of all files that are saved in data/ directory.
    '''
    indexed_files = __get_all_indexed_file_names__()
    files_in_data = os.listdir('data')
    return list(set(indexed_files).intersection(set(files_in_data)))

def __get_all_indexed_file_names__():
    '''
    Returns:
        list: A list of all file names based on data/index.json.
    '''
    files = []
    index = __load_index__()
    works = index['works']
    for work in works:
        for i, _ in enumerate(work['movements']):
            filename = __get_file_name__(work['key'], work['catalogue'], i + 1)
            files.append(filename)
    return files

def __download_file__(url, name, path, verbose=True):
    '''
    Downloads a file from the web and save it locally.

    Args:
        url (string): The remote URL where the file resides,
        name (string): Name given to the file,
        path (string): Directory where the file is to be saved,
    Returns
        bool, Exception: Whether the download was successful. If True, Exception will be none.
    '''
    assert path[-1] == '/', "The path must end with '/'."
    try:
        filename, headers = req.urlretrieve(url, path + name)
        if headers.get_content_type() != 'audio/midi':
          os.remove(path + name)
          raise Exception("Downloaded file content type is not audio/midi, daily limit may be reached.")
        if verbose:
            print("Downloaded to {}".format(filename))
        return True, None
    except Exception as error:
        if verbose:
            print("Failed to download {} with the following error: {}.".format(url, error))
        return False, error

def __get_file_name__(key, catalogue, mvmt):
    '''
    Returns a string containing the name for the given work. The file name has the following structure,
        key_catalogue_m.mid

    Args:
        key (string): The "key" attribute of the work,
        catalogue (string): The "catalogue attribute of the work,
        mvmt (int): The nth movement, where n starts at 1,

    Returns:
        string: Name given to the file.
    '''
    assert mvmt != 0, "Movement must start at 1, use n+1."
    out = key + "_" + catalogue +"_m" + str(mvmt) + ".mid"
    return out.replace(" ", "_")

def __load_index__():
    '''
    Returns a dictionary of all of the data compiled in data/index.json.

    Returns:
        dict: The data. See data/index.json for structure.
    '''
    with open('data/index.json') as file:
        content = file.read()
        index = json.loads(content)

    return index

if __name__ == '__main__':
  download_data()
