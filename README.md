# Nishika リチウムイオン電池の充電率予測  README
## i.使用したハードウェア（CPU/GPUのコア数/メモリ、ディスクサイズ、など）
<1>プロセッサ   
Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz  
、1992 Mhz

<3>実装RAM  
16.0GB

<4>ディスクサイズ  
476GB

## ii.バージョン情報を含む、使用したOS
Microsoft Windows 10 Home  
バージョン 1909 ビルド 18363.1556

Dockerバージョン情報 
Docker version 20.10.6, build 370c289

## iii.バージョン情報やインストール手順を含む、必要なサードパーティソフトウェア。これはDockerfileとして提供頂いても構いません
1)ディレクトリの最上層にDockerfileを記載した.   
2)Docker Desktop for Windowsをインストールし,Windows Power Shell内でDockerコマンドを使い環境を構築した.

## iv.乱数シードの情報
### 乱数の設定について
前処理,訓練,予測はそれぞれ,src内のファイルのpreprocess.py,train.py,predict.pyで行う.  
その乱数の設定は,それぞれのファイルのコード上部にあるseed_everything関数によって行っている.
seed値は42としている.
```
def seed_everything(seed=42):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)

seed_everything()
```

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

## vi.コードが実行時に前提とする条件
・vの手順5)にてコンテナに入った後,9)でlast_submission.csvが作成されるまでディレクトリ移動を
してはいけない.  
・必ず,vの手順2)では以下の様にファイルとフォルダを配置する必要がある.
data内部↓
![dataフォルダ内の配置](/fig1.png)
train内部↓
![trainフォルダ内の配置](/fig2.png)
・コンテナで使用できるメモリが6GB以外だと,preprocess.pyの実行中に途中で止まってしまう
ことがある.このような時は,windowsであれば,\Users\(ユーザー名)に.wslconfigというファイルを
作成し,その中のメモリの記述を6GB程度にする必要がある. (以下記述例)
```
[wsl2]
memory=6GB
```
・またwindows10 ProやMacで実行する場合にはメモリの設定は,docker homeのResourcesから行うことができる.

![.wslconfig](/fig3.png)

## vii.コードの重要な副作用
特になし.

















