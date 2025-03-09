# settings.py

import os

# プロジェクトの基本ディレクトリを設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# データ関連のパス
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
RAW_DATA_CSV_DIR = os.path.join(RAW_DATA_DIR, "sheet_sensor_csvdatas")
MAPPING_DATA_DIR = os.path.join(DATA_DIR, "mapping")
# データセットのパス
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
# 出力関連のパス
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_MAE_DIR = os.path.join(BASE_DIR, "outputs/mae")
# 途中の家庭を出力してる。
TEST_DIR = os.path.join(BASE_DIR, "tests")

# パラメータ
RATE = 500
RATE_15CH = 122.06
RATE_12CH = 147.10
# 決まっていること
TIME = 24  # 記録時間は24秒,これ変えるとMFER変換を変得ないといけない。
# その他の設定
DATASET_MADE_DATE = (
    "1226"  # packet_loss_data_[DATASET_MADE_DATE]となる部分。日付じゃなくてもいい。
)
DEBUG = True
