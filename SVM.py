import numpy as np
from tqdm import tqdm


def train(data, epochs, btz, lr=0.01, wd=0.01):
	data = np.array(data).astype('int')
	w = np.zeros(data.shape[-1])
	labels = 2*data[:, -1].copy()-1
	data[:, -1] = 1.0
	N = data.shape[0]

	for i in range(epochs):
		margin_corr = 0
		for j in tqdm(range(N//btz+1)):
			r = min((j+1)*btz, N)
			x = data[j*btz:r]
			y = labels[j*btz:r]
			tmp = np.dot(x, w)
			margin = y*tmp
			grad = -np.mean((margin <= 1)[..., None]*(2*wd*w-y[..., None]*x), 0)
			margin_corr += np.sum(margin > 1)
			w += lr*grad
		print(i, margin_corr/N)

	print(w)
	return w

def eval(w, data):
	data = np.array(data).astype('int')
	y = 2*data[:, -1].copy()-1
	data[:, -1] = 1.0
	corr = np.sum(y*np.dot(data, w) > 0)/data.shape[0]

	return corr