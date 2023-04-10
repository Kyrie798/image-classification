import os
import random
from shutil import copy

def mkdir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def main():
    random.seed(0)

    # 按照train ：val = 9 : 1的比例划分
    split_rate = 0.1

    data_root = './data'
    classes = [cls for cls in os.listdir(data_root)]

    # 保存训练集的文件夹
    train_root = './dataset/train'
    mkdir(train_root)
    for cls in classes:
        mkdir(os.path.join(train_root, cls))

    # 保存验证集的文件夹
    val_root = './dataset/val'
    mkdir(val_root)
    for cls in classes:
        mkdir(os.path.join(val_root, cls))

    for cls in classes:
        cls_path = os.path.join(data_root, cls)
        images = os.listdir(cls_path)
        num = len(images)
        eval_list = random.sample(images, k=int(num * split_rate)) # 从images列表中随机选取k个，组成新的列表
        for index, image in enumerate(images):
            if image in eval_list:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cls_path, image)
                new_path = os.path.join(val_root, cls)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cls_path, image)
                new_path = os.path.join(train_root, cls)
                copy(image_path, new_path)
            print('\r[{}] processing [{}/{}]'.format(cls, index + 1, num), end='')
        print()
    print('processing end!')

if __name__ == '__main__':
    main()
