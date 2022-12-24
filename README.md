# COVID-19-Prediction

Decision tree (0.760):

```
python -u main.py --downsampling 1.0
```

Bagging (0.770):

```
python -u main.py --skip_prob 0.5 0.9 --data_prob 0.3 --bagging --num_classifiers 16 --downsampling 1.0
```

AdaBoost (0.769):

```
python -u main.py --skip_prob 1.0 1.0 --boosting --num_classifiers 64 --downsampling 1.0 --data_prob 0.1
```

SVM (0.760):

```
python -u main.py --svm --epochs 500 --downsampling 1.0
```

Stacking (0.766):

```
python -u main.py --skip_prob 0.5 0.9 --data_prob 0.3 --stacking --epochs 500 --num_classifiers 16 --downsampling 1.0 --wd 0.0 --lr 0.01
```