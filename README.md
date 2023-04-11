# image-classification
This folder contains the basic image classification networks: [LeNet](https://github.com/Kyrie798/image-classification/tree/main/LeNet), [AlexNet](https://github.com/Kyrie798/image-classification/tree/main/LeNet), [VGG](https://github.com/Kyrie798/image-classification/tree/main/VGG), [GoogLeNet](https://github.com/Kyrie798/image-classification/tree/main/GoogLeNet), [ResNet](https://github.com/Kyrie798/image-classification/tree/main/ResNet)
[MobileNet](https://github.com/Kyrie798/image-classification/tree/main/MobileNet)  

You can download flower classification datasets from [flower](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) and put it into data  

## Prepare
```
# Before training, you should split the datasets into trian and val:
python split.py
```
## Train
```
# Start training with: 
python train.py
```

## Predict
```
# To test a image: 
python predict.py
```

