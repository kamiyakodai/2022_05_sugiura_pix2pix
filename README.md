# 2022_05_sugiura_pix2pix

### データセット
https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip
をCMPFacadeDatasets/facades/trainにダウンロード

https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip
をCMPFacadeDatasets/facades/valにダウンロード

datasetPutTogether.pyを実行

```sh
python datasetPutTogether.py
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
