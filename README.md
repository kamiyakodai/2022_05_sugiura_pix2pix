# 2022_05_sugiura_pix2pix

### 概要
このリポジトリは，CMPfacadeを用いたpix2pixの学習を行う．
3

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

12チャンネルで実行する場合

```sh
python CMPFacadeMultichannelconvert.py
```

### 実行方法

3チャンネルで実行する場合
```sh
python myPix2pix.py -c 3
```

12チャンネルで実行する場合
```sh
python myPix2pix.py -c 12
```
