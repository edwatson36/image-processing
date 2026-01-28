Package structure:
cub200_utils/
│
├── __init__.py
├── io_utils.py           # fetching and merging metadata
├── img_display_utils.py  # fetching and plotting images from file or tf batch
├── split_utils.py        # create test and validation splits, class balance + overlap checks
├── tf_utils.py           # preprocessing, augmentation, tf.Dataset pipeline
└── constants.py          # file names, column names, etc.


How to import into a notebook:
!pip install git+https://github.com/edwatson36/image-processing.git
import cub200_utils

See notebooks/preprocessing_using_cub200_Utils_lib.ipynb for full e2e use of the functions defined here and integration with a basic model.
