import pandas as pd
import os
import sys
import time

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import (
    DATA_DIR,
    BASE_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    TEST_DIR,
    RATE,
    RATE_16CH,
    TIME,
    DATASET_MADE_DATE,
)
from config.name_dic import select_name_and_date


class CSVReader_16ch:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self):
        files_found = []
        for filename in os.listdir(self.directory):
            if filename.startswith("db") and filename.endswith(".csv"):
                files_found.append(filename)
        return files_found

    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        df = pd.read_csv(file_path, header=None)
        print(f"ファイル {filename} を読み込みました。")
        # 読み込んだデータフレームの操作などを行う
        # ...
        print(df)
        return df

    def process_files(self):
        files_found = self.search_files()
        if len(files_found) > 0:
            df = self.read_csv_file(files_found[0])
            try:
                df = df.drop(columns=[17])
            except:  # 17列目が存在しない場合（Nan列がない場合）のエラーを無視
                pass
            print(df)
        else:
            print("指定した条件のCSVファイルは存在しません。")
            print("16ch")
            exit()
        return df


name, date = select_name_and_date()

db_dir_path = RAW_DATA_DIR + "/takahashi_test/{}/{}_{}_0/".format(name, name, date)
csv_reader_16ch = CSVReader_16ch(db_dir_path)
df_16ch = csv_reader_16ch.process_files()

# 最も右の列を連番データとして取得
seq = pd.to_numeric(df_16ch.iloc[:, -1], errors="coerce")

errors = []
i = 0
first = True

while i < len(seq):
    current = seq[i]

    if pd.isna(current):
        errors.append((i, None, "NaN値"))
        i += 1
        continue

    # 最初の値の場合：次に値が変わるまで進む
    if first:
        run_length = 1
        for j in range(i + 1, len(seq)):
            if seq[j] == current:
                run_length += 1
            else:
                break
        i += run_length
        first = False
        continue

    # 通常チェック：6個分同じであることを確認
    segment = seq[i : i + 6]
    if len(segment) < 6:
        errors.append((i, current, "残りが6個未満"))
        break

    if not (segment == current).all():
        mismatch = (segment != current).sum()
        errors.append((i, current, f"{mismatch}個が不一致（期待値: {current} x6）"))
        i += 1
        continue

    # 次の値のチェック
    if i + 6 < len(seq):
        next_val = seq[i + 6]
        expected_next = 0 if current == 255 else current + 1
        if next_val != expected_next:
            errors.append(
                (i + 6, next_val, f"期待値: {expected_next}（前の値: {current}）")
            )

    i += 6

# 結果出力
print(f"\n連番チェック結果：エラー数 = {len(errors)}\n")
for e in errors[:10]:
    print(f"Index: {e[0]}, 値: {e[1]}, 問題: {e[2]}")


# # 保存（オプション）
# pd.DataFrame(errors, columns=["Index", "Value", "Issue"]).to_csv(
#     "block_sequence_errors.csv", index=False
# )
# print("\n詳細は 'block_sequence_errors.csv' に保存されました。")
