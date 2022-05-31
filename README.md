# 2022_05_sugiura_pix2pix

### 概要
このリポジトリは，CMPfacadeを用いたpix2pixの学習を行う．
ラベル入力形式(3チャンネルと12チャンネル)と学習サンプル数の選択（ランダムに300, 350, 400, 450, 500枚）をコマンドラインオプションで指定．
学習に用いない画像すべて用いて，生成画像をFréchet inception distance (FID)で定量的に評価する．

### 実行環境
requirement.txtを参照

### データセット
https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip
をCMPFacadeDatasets/facades/trainにダウンロード

https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip
をCMPFacadeDatasets/facades/valにダウンロード

datasetPutTogether.pyを実行

```sh
python datasetPutTogether.py
```

ラベル画像を12チャンネルに変換

```sh
python CMPFacadeMultichannelconvert.py
```

### 実行方法

入力ラベル形式を3チャンネル，学習サンプル数を300枚で実行する場合
```sh
python myPix2pix.py -c 3 -s 300
```

入力ラベル形式を12チャンネル，学習サンプル数を500枚で実行する場合
```sh
python myPix2pix.py -c 12 -s 500
```
