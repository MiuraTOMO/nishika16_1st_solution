
# Nishika リチウムイオン電池の充電率予測 1st place solution  README
## データの準備







## v.モデルの学習から予測を行う際のソースコードの実行手順
### 1.環境構築
1)Documentsフォルダにzipから解凍した提出ファイルをおく.   
2)提出ファイルのdataフォルダ内に配布されたデータを配置する.   
(するとフォルダ内部はこのような配置になる.![dataフォルダ内の配置](/fig1.png))  
(train内部はこのような配置になっている.![trainフォルダ内の配置](/fig2.png))

2)次に,Windows Power Shellを用いて1)でおいた提出ファイルのパスに移動する.  
3)Windows Power ShellでDockerコマンドが実行できる状態で,以下のコマンドでDockerfileからイメージを作成.  
```
docker build -t nishika .
```
4)次にイメージが作成されたかを以下のコマンドで確認. (必ずしも必要ではない.)
```
docker images 
```
5)以下のコマンドからコンテナの起動し,コンテナ内に入る.
```
docker run --rm  -it --name nishika --mount type=bind,src="$(pwd)",dst=/nishika nishika bash
```
6)★データの前処理を行うためにsrcフォルダにあるpreprocess.pyを実行 (↓以下コード) 
#### (注意!!!! コンテナに入った後はディレクトリ移動は行わずに以下のコマンドを実行)
```
python ./src/preprocess.py
```
→./data/processedにnew_train_dfs.binaryfile,new_test_dfs.binaryfileが作成される.  
→new_train_dfs.binaryfileは訓練に使用する前処理後データである.  
→new_test_dfs.binaryfileは訓練済みモデルで予測に使用されるデータである.

7)★モデルの作成と訓練を行うためにsrcフォルダにあるtrain.pyを実行 (↓以下コード)
#### (注意!!!! 6)からディレクトリ移動を行わずに以下のコードを実行
```
python ./src/train.py
```
→modelsフォルダに予測に必要なモデルが保存される.予測の際はこのモデルが使われる.

8)★モデルの予測を行うためにsrcフォルダにあるpredict.pyを実行 (↓以下コード)
```
python ./src/predict.py
```
→modelsフォルダにあるモデルを使って6)で作成したnew_test_dfs.binaryfileに対して予測が行われる.  
→提出用ファイルはsubmisiionsフォルダに保存する.

9)これにて,submissionsフォルダの中身に提出したファイルlast_submission.csvが作成できる.

















