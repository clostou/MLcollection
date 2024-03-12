# # # # # # # # # # # # # # # # # # # # # # # #
#
#    前处理模块
#
# # # # # # # # # # # # # # # # # # # # # # # #

import os
import numpy as np
import re

from typing import List


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataBase')


def write(dataname: str, data: np.ndarray, label: list = None, tag: List = None) -> None:
    """
    Storage data set as file with readable format.

    :param dataname: name of the data set and output file
    :param data: data set (2-d numpy.ndarray object) to be written
    :param label: classification labels of all samples
    :param tag: description text of each attribute
    """
    n, m = data.shape
    has_label = bool(label)
    if not tag:
        tag = list(map(lambda i: 'Label' + str(i), range(1, n + 1))) + ['ClassLabel']
    header = f"{dataname}, {m}, {n}, {int(has_label)}\n{' '.join(tag)}\n"
    with open(os.path.join(data_dir, '%s.dat' % dataname), 'w', encoding='utf-8') as f:
        f.write(header)
        i = 0
        while i < m:
            if has_label:
                f.write(' '.join(data[: , i].astype(str)) + ' %s\n' % label[i])
            else:
                f.write(' '.join(data[: , i].astype(str)) + '\n')
            i += 1


def read(dataname: str) -> tuple:
    """
    Read data set from 'dataBase' folder.

    :param dataname: name of data set file
    :return: data, classification labels, attribute tags
    """
    with open(os.path.join(data_dir, '%s.dat' % dataname), 'r', encoding='utf-8') as f:
        try:
            header = f.readline()[: -1].split(',')
            info = list(map(int, header[1: 4]))
            tag = f.readline()[: -1].split()
        except:
            raise ValueError("Fail to read data file '%s.dat'" % dataname)
        data = np.zeros((info[0], info[1]), dtype=np.float32)
        label = []
        i = 0
        while i < info[0]:
            try:
                line = f.readline()[: -1].split()

                if info[2]:
                    data[i, :] = line[: -1]
                    label.append(line[-1])
                else:
                    data[i, :] = line
            except:
                raise ValueError("Invalid value in line %i" % (i + 3))
            i += 1
    try:
        label = list(map(int, label))
    except:
        pass
    print("-" * 32, "database: %s" % dataname, "count: %i" % len(label), "tags: %s" % tag, "-" * 32, sep='\n')
    return data.T, label, tag


def read_from_file(filepath: str, dataname: str, has_label: bool, tag: List[str] = None, density: float = 1.0) -> None:
    """
    Read data from external file and save to 'dataBase' folder.

    :param filepath: path of external data file
    :param dataname: name of data set
    :param has_label: if external data file contains classification label
    :param tag: (Optional) specify column tags in external data file explicitly
    :param density: (Optional) density of uniform sampling for smaller data set
    """
    print("Reading new data from file '%s' ..." % filepath)
    raw_data_list = []
    spliter = re.compile(r'[,\s]+')
    index = 1.
    with open(filepath, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            if line:
                if index >= 1.:
                    index -= 1.
                    raw_data_list.append(re.split(spliter, line[: -1]))
            else:
                break
            index += density
    try:
        raw_data = np.array(raw_data_list)
        m, n = raw_data.shape
    except:
        raise ValueError("Invalid shape of data in file '%s'" % filepath)
    if has_label:
        n = n - 1
    data_list = []
    tag_list = []
    for i in range(n):
        if tag:
            column_tag = tag[i].replace(' ', '_')
        else:
            column_tag = '特征%s' % (i + 1)
        try:
            column = raw_data[: , i].astype(np.float32)
            if column.var() != 0:
                data_list.append(column)
                tag_list.append(column_tag)
        except:
            column = raw_data[:, i]
            cls = np.unique(column)
            if len(cls) > 1:
                for s in cls:
                    column_onehot = (column == s).astype(np.float32)
                    data_list.append(column_onehot)
                    tag_list.append('%s-%s' % (column_tag, s))
    if len(data_list) > n:
        print("One-hot encoding: %i -> %i" % (n, len(data_list)))
    if has_label:
        column = raw_data[:, -1]
        cls = np.unique(column)
        if len(cls) == 2:
            label = np.ones(m, dtype=np.int32)
            label[column == cls[1]] = -1
        else:
            label = column
        label = label.tolist()
        if tag:
            tag_list.append(tag[-1].replace(' ', '_'))
        else:
            tag_list.append('类别')
        print("Classification count: %i" % len(cls))
    else:
        label = None
    write(dataname, np.stack(data_list, axis=1).T, label, tag_list)
    print(f"Totally {m} simples saved in database '{dataname}'.")


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Nomalize each feature of data to range 0~1

    :param data: data to process
    :return: normalized data
    """
    return data / np.tile(np.max(data, axis=data.ndim - 1, keepdims=True), data.shape[-1])


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Adjust mean to 0 and variance to 1 on each feature

    :param data: data to process
    :return: standardized data
    """
    data = data.T
    var = data.var(axis=0)
    if data.ndim == 2:
        ind = np.nonzero(var == 0)[0]
        data = np.delete(data, ind, axis=1)
        var = np.delete(var, ind)
    return ((data - data.mean(axis=0)) / np.sqrt(var)).T


if __name__ == '__main__':
    write('test', np.array([[1, 2, 0], [-1, 1, 2]]), [1, 1, -1])
    data, label, tag = read('watermelon_3.0')
    data_s = standardize(data)


