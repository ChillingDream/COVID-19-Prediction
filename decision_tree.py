import numpy as np


def entropy(x, weights):
    tot = weights.sum()
    p0 = weights[x == 0].sum() / tot + 1e-10
    p1 = weights[x == 1].sum() / tot + 1e-10
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


class Node:
    def __init__(self):
        self.key = None
        self.label = None
        self.children = {}
    
    def construct(self, data, weights, skip_prob, used_feature=[]):
        cnt = data['label'].value_counts()
        if len(cnt) == 1:
            self.label = cnt.index[0]
        elif len(data.columns) == len(used_feature):
            if cnt[True] > cnt[False]:
                self.label = True
            else:
                self.label = False
        else:
            H = entropy(data['label'], weights)
            max_IGR = -1e10
            total = weights.sum()
            tot_num = len(data.columns) - len(used_feature)
            cnt_num = 0
            for name in data.columns:
                if name in used_feature:
                    continue
                cnt_num += 1
                if np.random.uniform() < skip_prob[0] - 1e-10 and tot_num > cnt_num:
                    continue
                CH = 0
                IV = 0
                for value in data[name].unique():
                    subdata = data[data[name] == value]
                    subweights = weights[data[name] == value]
                    p = subweights.sum() / total + 1e-10
                    IV -= p * np.log2(p)
                    CH += p * entropy(subdata['label'], subweights)
                IGR = (CH - H) / IV
                if IGR > max_IGR:
                    max_IGR = IGR
                    self.key = name
            for value in data[self.key].unique():
                subdata = data[data[self.key] == value]
                subweights = weights[data[self.key] == value]
                self.children[value] = Node()
                self.children[value].construct(subdata, subweights, [skip_prob[0] * skip_prob[1], skip_prob[1]], used_feature + [self.key])
        
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
    def __init__(self, data, weights, skip_prob):
        self.root = Node()
        self.root.construct(data, weights, skip_prob, ['label'])
    
    def predict(self, x):
        node = self.root
        while node.label is None:
            node = node.next(x[node.key])
        return node.label
