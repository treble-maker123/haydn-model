import os
import uuid
from time import time, strftime
from datetime import datetime

OUTPUT_PATH = os.getcwd() + "/outputs"
SAMPLE_PATH = os.getcwd() + "/samples"

def get_unique_id():
    return uuid.uuid4().hex.upper()[0:6]

def get_formatted_time():
    return datetime.now().strftime("%m-%d_%H-%M-%S")

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
