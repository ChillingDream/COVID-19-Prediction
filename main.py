import argparse
import pandas as pd
import random
import numpy as np
from scipy import stats
from dataset import CovidDataset
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = argparse.ArgumentParser(description='OGBL')
parser.add_argument('--bagging', action='store_true')
parser.add_argument('--boosting', action='store_true')
parser.add_argument('--downsampling', type=float, default=0.001)
parser.add_argument('--skip_prob', type=float, default=[0.5, 0.8], nargs='+')
parser.add_argument('--data_prob', type=float, default=0.8)
parser.add_argument('--num_classifiers', type=int, default=8)
args = parser.parse_args()

def run_decision_tree(train_ds, test_ds, weights, args):
    train_ds = pd.concat(train_ds, axis=1).T
    tree = DecisionTree(train_ds, weights, args.skip_prob)
    corr = 0
    for x in tqdm(test_ds):
        corr += tree.predict(x) == x['label']
    print(corr / len(test_ds))

    return tree

if __name__ == "__main__":
    print(args)
    ds = CovidDataset('Covid Data.csv')
    plain_train_ds, test_ds = train_test_split(ds, test_size=1000, random_state=42)

    train_ds = random.sample(plain_train_ds, int(len(plain_train_ds)*args.downsampling))
    if args.bagging:
        clf = []
        tot_num = len(train_ds)
        num = int(tot_num*args.data_prob)
        for i in range(args.num_classifiers):
            ds = random.sample(train_ds, num)
            weights = np.ones(len(ds))
            tree = run_decision_tree(ds, test_ds, weights, args)
            clf.append(tree)

        corr = 0
        for x in tqdm(test_ds):
            vote = []
            for i in range(args.num_classifiers):
                vote.append(clf[i].predict(x))
            corr += stats.mode(vote)[0][0] == x['label']
        print
        print('bagging res:', corr / len(test_ds))
    elif args.boosting:
        weights = np.ones(len(train_ds))
        clf_weights = np.ones(args.num_classifiers).tolist()
        clf = []
        for idx in range(args.num_classifiers):
            tree = run_decision_tree(train_ds, test_ds, weights, args)
            clf.append(tree)
            eps = 0
            total = weights.sum()
            cnt = 0
            incorrect = np.zeros(len(train_ds))
            for i in range(len(train_ds)):
                x = train_ds[i]
                incorrect[i] = tree.predict(x) != x['label']
                eps += weights[i] * incorrect[i]
                incorrect[i] = 2 * incorrect[i] - 1
            eps /= total
#            gamma = 1 - 2 * eps
            alpha = 0.5 * np.log((1 - eps) / eps)
            clf_weights[idx] = alpha
            weights = weights*np.exp(alpha*incorrect)
            print(alpha)
        corr = 0
        for x in tqdm(test_ds):
            pred = 0
            for i in range(args.num_classifiers):
                pred += clf_weights[i] * (2 * clf[i].predict(x) - 1)
            corr += (pred > 0) == x['label']
        
        print('boosting res:', corr / len(test_ds))
    else:
        weights = np.ones(len(train_ds))
        tree = run_decision_tree(train_ds, test_ds, weights, args)