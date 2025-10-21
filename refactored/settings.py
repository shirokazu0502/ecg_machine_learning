import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
RAW_DATA_CSV_DIR = os.path.join(RAW_DATA_DIR, "sheet_sensor_csvdatas")
MAPPING_DATA_DIR = os.path.join(DATA_DIR, "mapping")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_MAE_DIR = os.path.join(BASE_DIR, "outputs/mae")
TEST_DIR = os.path.join(BASE_DIR, "tests")

RATE = 500
RATE_15CH = 122.06
RATE_16CH = 122.06
RATE_12CH = 147.10
TIME = 24
DATASET_MADE_DATE = "0602"
DEBUG = True
