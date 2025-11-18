# ecg_machine_learning

## 概要 (Overview)

このプロジェクトは、胸部に貼り付ける16チャンネルECG電極シートから得られるデータに対し、被験者ごとの心臓電気軸を基準とした回転補正を適用するためのプログラム群です。電極シートの物理的な貼り付けズレを補正し、より標準化された心電図データを生成することを目的とします。

処理の中核として、物理16電極から仮想的な9電極を生成し、その仮想電極を基準に電気軸の計算と回転補正を行うことで、安定性と精度を向上させています。

## 処理フロー (Processing Workflow)

データ処理は、以下の3つのスクリプトを順番に実行することで完結します。

1.  **`create_virtual_dataset.py` の実行**
    *   **目的**: 物理16チャンネルの生データセットから、回転補正を適用する前の「仮想9チャンネルデータセット」を作成します。
    *   **入力**: `data/processed/for_best_resample/{subject_name}/` にある物理16chデータセット (`dataset_*.csv`)。
    *   **出力**: `data/processed/virtual_electrode_dataset/{subject_name}/` に、各心拍に対応する仮想9chデータセット (`dataset_*.csv`) を生成します。

2.  **`create_virtual_electrode_axis.py` の実行**
    *   **目的**: 被験者の心臓電気軸を計算します。
    *   **入力**: `data/processed/for_best_resample/{subject_name}/` にある物理16chデータセット。
    *   **処理**:
        1.  全心拍データを平均化し、平均波形を生成します。
        2.  平均波形から仮想9電極の波形を生成します。
        3.  `neurokit2` ライブラリを用いてQRS波を検出し、仮想電極のベクトルから電気軸の角度を計算します。
    *   **出力**: `data/processed/virtual_electrode_dataset/{subject_name}/` に、電気軸の角度やQRS情報を含む `virtual_axis.csv` を生成します。

3.  **`apply_virtual_rotation_correction.py` の実行**
    *   **目的**: 計算された電気軸を用いて、データに回転補正を適用し、最終的なデータセットを生成します。
    *   **入力**:
        1.  `data/processed/for_best_resample/{subject_name}/` にある物理16chデータセット。
        2.  上記2で生成された `virtual_axis.csv`。
    *   **出力**: `data/processed/rotated_datasets/{subject_name}_virtual_rotated/` に、回転補正済みの仮想9chデータセット (`dataset_*.csv`) を生成します。

## 各スクリプトの詳細 (Scripts Details)

### `create_virtual_dataset.py`
- **役割**: 物理16chデータセットを仮想9chデータセットに変換します。各心拍ファイルが個別に変換されます。
- **列の順序**: 出力されるCSVファイルの列は `Time`, `仮想9ch`, `医療12誘導` の順に整列されます。

### `create_virtual_electrode_axis.py`
- **役割**: 被験者一人分のデータ全体から、単一の心臓電気軸を算出します。
- **特徴**:
    - 内部で全心拍を平均化し、ノイズに強い安定した電気軸を計算します。
    - QRS検出には `neurokit2` を使用し、精度を向上させています。
    - 計算結果の `virtual_axis.csv` には、角度だけでなく、計算に使用したQRS波の位置と振幅も含まれます。

### `apply_virtual_rotation_correction.py`
- **役割**: プロジェクトの核心部分。電気軸に基づき、波形に多様性を持たせた回転補正を適用します。
- **補正ロジック (2段階補間)**:
    1. **高解像度マップ作成**: 物理16電極の信号を元に、`griddata` を用いて17x17=289点の高解像度な「電位マップ」を中間生成します。
    2. **最終補間**:
        - 仮想9電極の初期座標（物理グリッド上の絶対座標 (0.5, 2.5) など）を、電気軸の角度とシート中心(1.5, 1.5)を基準に回転させ、最終的な目標座標を計算します。
        - この目標座標における電位を、上記1で作成した**高解像度マップ**から `griddata` で補間して読み取ります。
- **利点**: この2段階補間により、補間元の情報量が増え、補正後の波形がより滑らかで多様性を持つようになります。

## 実行方法 (How to Run)

現在、上記3つのスクリプトは、特定の被験者名でパスがハードコードされています。新しい被験者のデータ処理を行うには、各スクリプトの `if __name__ == "__main__":` ブロック内にある以下の変数を修正する必要があります。

```python
# 例: create_virtual_dataset.py の場合
if __name__ == "__main__":
    ...
    # この値を対象の被験者ディレクトリ名に変更する
    subject_name_full = "asano_0714_0.8s" 
    ...
    main(args)
```

処理したい被験者名（例: `kanda_0807_0.8s`）に合わせてこの `subject_name_full` を変更し、**処理フロー**に記載の順番通りに3つのスクリプトを実行してください。