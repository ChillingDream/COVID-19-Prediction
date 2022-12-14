import pandas as pd
from dataset import CovidDataset
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ds = CovidDataset('Covid Data.csv')
train_ds, test_ds = train_test_split(ds, test_size=1000, random_state=42)

def run_decision_tree(train_ds, test_ds):
    train_ds = pd.concat(train_ds, axis=1).T
    tree = DecisionTree(train_ds)
    corr = 0
    for x in tqdm(test_ds):
        corr += tree.predict(x) == x['label']
    print(corr / len(test_ds))

run_decision_tree(train_ds, test_ds)