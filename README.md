# インストール方法


- DiceDataset_20181214.zipは私の環境では```~/DATA/NicoTechDice/DiceDataset```に展開しました。
- ファイルやフォルダはmake_data_directory.pyを使用して以下のように配置してください。
```
DiceDataset/
    train/
        1/
            00001.bmp
            00002.bmp
            ...
        2/
            00001.bmp
            00002.bmp
            ...
    valid/
        1/
            00001.bmp
            00002.bmp
            ...
        2/
            00001.bmp
            00002.bmp
            ...
    test/
        1/
            00001.bmp
            00002.bmp
            ...
        2/
            00001.bmp
            00002.bmp
            ...
            ...
```
上記のように配置できたら、DiceDatasetのルートにある1,2,3,4,5,6などのディレクトリは削除して大丈夫です。（train,valid,testディレクトリは使います）


## インストール
```bash
pip install pillow
pip install keras
pip install tensorflow

```

学習は、```python cnn.py```を実行することでできます。


## 参考文献
- https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
- https://keras.io/preprocessing/image/
