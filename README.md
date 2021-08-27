
# Nishika リチウムイオン電池の充電率予測 1st place solution  README


## 1.データの準備
TrainデータとTestデータは[コンペ公式サイト](https://www.nishika.com/competitions/16/data)からダウンロードして、以下のようにファイルを格納してください

```
├── data_explanation.xlsx
├── processed
├── sample_submission.csv
├── test.csv
└── train
    ├── n10degC_Cycle_1_Pan18650PF.csv
    ├── n10degC_NN_Pan18650PF.csv
    ├── n10degC_US06_Pan18650PF.csv
    ├── n20degC_Cycle_1_Pan18650PF.csv
    ├── n20degC_NN_Pan18650PF.csv
    ├── n20degC_US06_Pan18650PF.csv
    ├── p0degC_Cycle_1_Pan18650PF.csv
    ├── p0degC_NN_Pan18650PF.csv
    ├── p0degC_US06_Pan18650PF.csv
    ├── p10degC_Cycle_1_Pan18650PF.csv
    ├── p10degC_NN_Pan18650PF.csv
    ├── p10degC_US06_Pan18650PF.csv
    ├── p25degC_Cycle_1_Pan18650PF.csv
    ├── p25degC_NN_Pan18650PF.csv
    └── p25degC_US06_Pan18650PF.csv


```
## 2.フォルダ構成
```
├── Dockerfile             -環境構築
├── directory_structure.txt
├── models　　　　　　　　　-学習済みモデル
├── requirements.txt       -必要なライブラリを明記
├── run.sh                 -実行コマンド
├── setting.json           -環境変数の設定
├── src
│   ├── predict.py         -予測ファイルの作成
│   ├── preprocess.py      -前処理
│   └── train.py           -モデルの訓練
└── submissions            -提出用ファイル格納用
```

## 3.環境構築
1)任意のパスに本レポジトリを格納する。

2)dataフォルダ内に1.データの準備の様にデータを格納する。

3)次に,Dockerコマンドが実行できる任意のターミナルを用いて1)のパスに移動する。

4)以下のコマンドでDockerfileからイメージを作成。
```
docker build -t nishika .
```

5)次にイメージが作成されたかを以下のコマンドで確認。(必ずしも必要ではない。)
```
docker images 
```

6)以下のコマンドからコンテナの起動し、コンテナ内に入る。
```
docker run --rm  -it --name nishika --mount type=bind,src="$(pwd)",dst=/nishika nishika bash
```

7)★データの前処理を行うためにsrcフォルダにあるpreprocess.pyを実行する。 (↓以下コード) 
#### (注意!!!! コンテナに入った後はディレクトリ移動は行わずに以下のコマンドを実行)
```
python ./src/preprocess.py
```
→./data/processedにnew_train_dfs.binaryfile,new_test_dfs.binaryfileが作成される。
→new_train_dfs.binaryfileは訓練に使用する前処理後データである。
→new_test_dfs.binaryfileは訓練済みモデルで予測に使用されるデータである。

8)★モデルの作成と訓練を行うためにsrcフォルダにあるtrain.pyを実行する。 (↓以下コード)
#### (注意!!!! 6)からディレクトリ移動を行わずに以下のコードを実行
```
python ./src/train.py
```
→modelsフォルダに予測に必要なモデルが保存される.予測の際はこのモデルが使われる。

9)★モデルの予測を行うためにsrcフォルダにあるpredict.pyを実行する。 (↓以下コード)
```
python ./src/predict.py
```
→modelsフォルダにあるモデルを使って6)で作成したnew_test_dfs.binaryfileに対して予測が行われる。  
→提出用のcsvはsubmisiionsフォルダに保存される。

### 謝辞
今回このような興味深い自然科学に関するコンペを開催してくださったNishika様、
自然科学のイロハを教えてくださった恩師、機械学習の知識と技術を教えてくださった
現指導教官に感謝申し上げます。






