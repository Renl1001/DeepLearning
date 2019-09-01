import os
from shutil import copyfile


def copy_file(max_num):
    """复制部分数据到新的文件夹中，降低数据量

    Arguments:
        max_num {int} -- 每个类别的最大数量
    """
    MAX_NUM = max_num

    path = 'THUCNews'
    dst_path = 'mini_News'
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

    for item in os.listdir(path):
        # print(item)
        class_path = os.path.join(path, item)
        dst_class_path = os.path.join(dst_path, item)
        if not os.path.isdir(dst_class_path):
            os.mkdir(dst_class_path)
        if os.path.isdir(class_path):
            num = 0
            for file in os.listdir(class_path):
                num += 1
                if num > MAX_NUM:
                    break
                file_path = os.path.join(class_path, file)
                dst_file_path = os.path.join(dst_class_path, file)
                copyfile(file_path, dst_file_path)


def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t',
                                                  '').replace('\u3000', '')


def save_file(dirname):
    """
    将多个文件的数据整合到三个txt文件中

    Arguments:
        dirname {str} -- 目录名
    """
    f_train = open('cnews/cnews.train.txt', 'w', encoding='utf-8')
    f_test = open('cnews/cnews.test.txt', 'w', encoding='utf-8')
    f_val = open('cnews/cnews.val.txt', 'w', encoding='utf-8')
    for category in os.listdir(dirname):  # 分类目录
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)
            if count < 800:
                f_train.write(category + '\t' + content + '\n')
            elif count < 900:
                f_test.write(category + '\t' + content + '\n')
            else:
                f_val.write(category + '\t' + content + '\n')
            count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()


copy_file(1000)
save_file('mini_News')
