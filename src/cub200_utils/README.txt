Package structure:
cub200_utils/
│
├── __init__.py
├── io_utils.py           # fetching and merging metadata
├── validation_utils.py   # class balance + overlap checks
├── tf_utils.py           # preprocessing, augmentation, tf.Dataset pipeline
└── constants.py          # file names, column names, etc.


How to import into a notebook:
from cub200_utils import *

See notebooks/cub_200_2011_preprocessing.ipynb for full e2e use of the functions defined here and integration with a basic model.


Disclaimers:
Still a work in progress with more rigorous unit testing to be done and some deprecation warnings to resolve.
