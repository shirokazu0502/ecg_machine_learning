# ECG_PROJECT（薄膜電極アレイ心電計）

# DEMO
![alt text](/images/overview.png)
![alt text](/images/Flexible%20sheet.png)



# Requirement

* numpy
* pandas
* matplotlib
* gpuが使用できるサーバー

など，適宜インストール
# Installation
192.168.29.72は全てインストールされている。
```
pip install numpy
```
などで適宜使用するサーバにダウンロードする。（本当は仮想環境などでプロジェクトごとに環境分けた方がいい）



# Directory
全体のディレクトリ構造
```
├── README.md
├── config #設定関連
│   ├── __init__.py
│   ├── __pycache__
│   ├── name_dic.py #データセットの人物と日付
│   └── settings.py #サンプリングレートや変数
├── data
│   ├── mapping #体表面電位分布の時間変化プロットのデータ
│   ├── processed #機械学習のデータセット
│   └── raw #生データ
├── images #READMEの画像
│   ├── Flexible sheet.png
│   └── overview.png
├── outputs
│   ├── Errors #振幅の誤差MAE
│   ├── Scatters #T波ピーク位置の誤差MAE
│   ├── figs_newref #出力波形の誤差
│   ├── mae #振幅の誤差MAE
│   └── pqrst_nkmodule_since0120_cwt #
├── requirements.txt
├── src
│   ├── dataset_creation #データセットの作成関連コード
│   ├── machine_learinig #機械学習関連コード
│   ├── mapping_15ch #体表面電位分布プロット関連コード
│   └── sheet_sensor #シートセンサ計測関連コード
└── tests
│   └── raw_datas_test　#生波形を試しにプロットする結果
└── case_data
```
# ①Sheet_sensor
シートセンサでの計測関連のディレクトリ
```
├── src/sheet_sensor
    ├── copy_file.py #/data/rawにコピーするコード
    ├── new_16ch.bat #read_sensor16ch202306.pyとcopy_file.pyを呼び出すコード
    ├── packet_check.py #パケット落ちを確認するコード
    ├── packet_check_sensor.py
    ├── plot.py #描画するコード
    └── read_sensor16ch202306.py #計測するコード
```
# ②Dataset_creation
データを前処理し，データセット作成の関連ディレクトリ
```
├── src/dataset_creation
    ├── Make_dataset_0120.py #データセット作成コード。/data/processed/[dataset名]に出力。データセット名はconfig/settingで設定されてる
    ├── mfer_to_12ch.py #12誘導心電計のMFERデータを読み出すコード。これはバイナリ解読して自作した。24秒間のデータはこのままで変換できる。
```
# ③Machine_learning
機械学習を行うコード
```
├── src/machine_learning
    ├── __pycache__
    │   ├── Dataset.cpython-38.pyc
    │   └── models.cpython-38.pyc
    ├── models.py#モデル構造
    ├── output_data_pqrst_v3.py#T波ピーク位置の評価するためのコード
    ├── simple.sh #シンプルなVAEだけ実行するコード
    ├── vae_cross_v2_5.pth #学習済みの重み。学習回すたびに更新される
    └── vae_goto_val_pt.py#VAEのメインファイル。
```
# ④Mapping
体表面電位をプロットするコード
![alt text](/images/mapping.png)
佐藤に対してデータを12回位置変えて計測した
一拍分0.8秒を10拍分加算平均
データは/data/mappingに配置
```
├── src/mapping
    ├── R-wave.png
    ├── T-wave.png
    ├── heatmap_animation.mp4 #ヒートマップ
    ├── heatmap_animation_aaaa.mp4
    ├── heatmap_animation_org.mp4
    ├── heatmap_animation_t.mp4
    ├── heatmap_animation_twave.mp4
    ├── mapping.py #メインファイル。
    ├── org_wave_check_sato.py
    ├── sato_all.csv
    └── wave.mp4
```

# Usage

①シートセンサ計測コード実行，フォルダ作成してコピー
```
new_16ch.bat
```
②データセットの作成実行コード
```
python3 mfer_to_12ch.py#mferをcsvに変換
python3 Make_dataset.py
```
③機械学習
```
sh simple.sh
```
or
```
sh 2023_0121.sh
```

# ⑤Case data
無線計測回路のケースデータ。大阪大学にもこのケースに入れたセンサを送付している。
データは/case_dataに配置。
Fusion 360 で編集できる .f3dファイル
Fusion 360 でエクスポートする .3mfファイルをFlash Printで読み込む
Flash Printでサポート材などを設定し，3Dプリンタに送信し印刷

# Author
作成情報を列挙する
* 作成者:後藤
* 年度:2022-2023
* E-mail: g.yusaku.0119@icloud.com
