
import pre
import regress
import reduce
import neural
import post

import numpy as np
#import numpy.linalg as la
#from matplotlib import pyplot as plt


flag = (0, 0, 0, 1, 0)


if flag[0]:
    '''
    Test of Logit Regression
    '''
    dataname = 'watermelon_3.0alpha'
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
    l = regress.Logit(data, label)
    l.train()
    post.item_print('beta vector', l.beta, newline=True)
    post.plot(data, label, line=l.beta, title=dataname, tag=tag)
    #l.classify(data)


if flag[1]:
    '''
    Test of Linear Discriminant Analysis
    '''
    dataname = 'watermelon_3.0alpha'
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
    l = reduce.lda(data, label)
    l.train()
    post.item_print('W matrix', l.W, newline=True)


if flag[2]:
    '''
    Test of Recursive Least Squares
    '''
    dataname = 'watermelon_3.0alpha'
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
    # 重映射样本标签
    label = list(map(lambda x: 2 * x - 1, label))
    # 以下4行：随机打乱数据集
    randInd = np.arange(len(label))
    np.random.shuffle(randInd)
    data = data[: , randInd]
    label = np.array(label)[randInd].tolist()
    l = regress.RLS(data, label)
    l.train()
    post.item_print('w matrix', l.w, newline=True)
    post.plot(data, label, line=l.w, title=dataname, tag=tag)


if flag[3]:
    '''
    Test of BP Algorithm
    
        易陷入局部最优。考虑对错分样本单独学习，而不是乱序对所有样本进行学习
    '''
    dataname = 'watermelon_3.0'
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
    n = neural.MFF1(data, label, hidden_neuron_count=4)
    print(0.5 * np.mean(np.linalg.norm(n.label.T - n.classify(data), axis=0)))
    n.train(200, True)
    #post.item_print('hidden-layyer w:', n.hidden_layer.w, True)
    #post.item_print('hidden-layyer b:', n.hidden_layer.b, True)
    #post.item_print('output-layyer w:', n.output_layer.w, True)
    #post.item_print('output-layyer b:', n.output_layer.b, True)
    print(0.5 * np.mean(np.linalg.norm(n.label.T - n.classify(data), axis=0)))


if flag[4]:
    '''
    Test of SOM Network
    '''
    dataname = 'watermelon_3.0alpha'
    data, label, tag = pre.read(dataname)
    print('database: %s' % dataname, 'count: %i' % len(label), 'tags: %s' % tag, sep='\n')
    n = neural.SOM(data, network_size=10)
    n.train(cycle_count=2000)
    post.mat_scatter(n.classify(data), label)