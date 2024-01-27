# # # # # # # # # # # # # # # # # # # # # # # #
#
#    前处理模块
#
# # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np


def write(dataname: str, data: np.ndarray, label=[], tag=[]):
    """
    Storage data set as file with readable format.

    :param dataname: name of the data set and output file
    :param data: data set (2-d numpy.ndarray object) to be written
    :param label: classification labels of all samples
    :param tag: description text of each attribute
    """
    m, n = data.shape
    has_label = bool(label)
    if not tag:
        tag = list(map(lambda i: 'Label' + str(i), range(1, n + 1))) + ['ClassLabel']
    header = f"{dataname}, {m}, {n}, {int(has_label)}\n{' '.join(tag)}\n"
    with open('.\\dataBase\\%s.dat' % dataname, 'w', encoding='utf-8') as f:
        f.write(header)
        i = 0
        while i < m:
            if has_label:
                f.write(' '.join(data[: , i].astype(str)) + ' %i\n' % label[i])
            else:
                f.write(' '.join(data[: , i].astype(str)) + '\n')
            i += 1


def read(dataname: str):
    """
    Read data set from file.

    :param dataname: name of data set file
    :return: data, classification labels, attribute tags
    """
    with open('.\\dataBase\\%s.dat' % dataname, 'r', encoding='utf-8') as f:
        try:
            header = f.readline()[: -1].split(',')
            info = list(map(int, header[1: 4]))
            tag = f.readline()[: -1].split()
        except:
            raise ValueError("Fail to read data file '%s.dat'" % dataname)
        data = np.zeros((info[0], info[1]), dtype=np.float)
        label = []
        i = 0
        while i < info[0]:
            try:
                line = f.readline()[: -1].split()
                if info[2]:
                    data[i, : ] = line[: -1]
                    label.append(int(line[-1]))
                else:
                    data[i, : ] = line
            except:
                raise ValueError('Invalid value in line %i' % (i + 3))
            i += 1
    return data.T, label, tag


def normalize(data: np.ndarray):
    """
    Nomalize each feature of data to range 0~1

    :param data: data to process
    :return: normalized data
    """
    return data / np.tile(np.max(data, axis=data.ndim - 1, keepdims=True), data.shape[1])


if __name__ == '__main__':
    write('test', np.array([[1, 2, 0], [-1, 1, 2]]), [1, 1])
    data, label, tag = read('watermelon_3.0')
