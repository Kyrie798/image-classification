import os
import random
import json

def read_split_data(root, val_rate):
    random.seed(0)

    # 生成class_indices.json文件
    classes = [cla for cla in os.listdir(root)]
    classes.sort()
    classes_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((value, key) for key, value in classes_indices.items()), indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)
    
    train_images_path = [] # 训练集图像路径
    train_label = [] # 训练集索引
    val_images_path = []
    val_label = []
    every_class_num = [] # 每个类别的样本总数

    for cla in classes:
        cla_path = os.path.join(root, cla)
        # 保存所有图像路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
        images.sort()
        # 保存类别对应的索引
        image_class = classes_indices[cla]
        # 记录类别样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证集
        val_path = random.sample(images, k=int(len(images) * val_rate))
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    return train_images_path, train_label, val_images_path, val_label
