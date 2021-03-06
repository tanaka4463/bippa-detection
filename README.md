# bippa-detection

faster RCNNを使ってビッパの検出を行うプログラムです。
ポケモンスナップで100枚程度のビッパを撮影し、学習を行っております。
labelmeを使ってアノテーションを実施し、出力されたjsonファイルに従って、データセットを作成しました。

## Requirement
python 3.6.10  
torch: 1.4.0  
sklearn: 0.24.1  
torchvision: 0.5.0  
numpy: 1.19.2  
matplotlib: 3.3.3  
cv2: 4.4.0  
tqdm: 4.56.0  
PIL: 8.0.1  
pandas: 1.1.5 

## Demo
![bippa](https://github.com/tanaka4463/bippa-detection/blob/main/img/2021053000371400-194D89293F260C6893CF3FBF65B93019.jpg)
![detection](https://github.com/tanaka4463/bippa-detection/blob/main/img/bippa.jpg)


## Usage
`$ python main.py`  
上記プログラムで学習させた後、inference()の実行で推論を行います。

## Note
学習モデルが25MBを超えているため、アップロードできておりません。  
ご承知おきください。
