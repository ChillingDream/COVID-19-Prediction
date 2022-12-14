import numpy as np


def entropy(x):
    p0 = (x == 0).sum() / len(x) + 1e-10
    p1 = (x == 1).sum() / len(x) + 1e-10
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


class Node:
    def __init__(self):
        self.key = None
        self.label = None
        self.children = {}
    
    def construct(self, data, used_feature=[]):
        cnt = data['label'].value_counts()
        if len(cnt) == 1:
            self.label = cnt.index[0]
        elif len(data.columns) == len(used_feature):
            if cnt[True] > cnt[False]:
                self.label = True
            else:
                self.label = False
        else:
            H = entropy(data['label'])
            max_IGR = -1e10
            total = len(data)
            for name in data.columns:
                if name in used_feature:
                    continue
                CH = 0
                IV = 0
                for value in data[name].unique():
                    subdata = data[data[name] == value]
                    p = len(subdata) / total + 1e-10
                    IV -= p * np.log2(p)
                    CH += p * entropy(subdata['label'])
                IGR = (CH - H) / IV
                if IGR > max_IGR:
                    max_IGR = IGR
                    self.key = name
            for value in data[self.key].unique():
                subdata = data[data[self.key] == value]
                self.children[value] = Node()
                self.children[value].construct(subdata, used_feature + [self.key])
        
    def next(self, v):
        if v in self.children:
            return self.children[v]
        keys = sorted(list(self.children.keys()))
        for key in keys:
            if v <= key:
                return self.children[key]
        else:
            return self.children[keys[-1]]
            

class DecisionTree:
    def __init__(self, data):
        self.root = Node()
        self.root.construct(data, ['label'])
    
    def predict(self, x):
        node = self.root
        while node.label is None:
            node = node.next(x[node.key])
        return node.label
