import os

mode = os.environ.get('MODE', 'train').lower()

from .args import *

os.environ["MODEL_NAME"] = args.model
os.environ['AUG_NAME'] = args.model

from .path import *
from .logger import *
from .dataloader import *

if mode == 'test':
    from .metrics import *