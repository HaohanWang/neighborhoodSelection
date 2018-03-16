__author__ = 'Haohan Wang'

import numpy as np

def loadData_naive():
    text = [line.strip() for line in open('../data/Data.csv')]
    texts = text[0].split('\r')

    data = []
    for line in texts[1:]:
        items = line.split(',')
        l = [float(a) for a in items[3:-7]]
        data.append(l)

    data = np.array(data)

    return data