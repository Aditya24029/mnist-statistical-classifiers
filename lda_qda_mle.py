import os
import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Load MNIST Files

def load_images(path):

    f = open(path, "rb")

    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))

    data = np.frombuffer(f.read(), dtype=np.uint8)

    data = data.reshape(num, rows*cols)

    f.close()

    return data


def load_labels(path):

    f = open(path, "rb")

    magic, num = struct.unpack(">II", f.read(8))

    labels = np.frombuffer(f.read(), dtype=np.uint8)

    f.close()

    return labels


train_X = load_images("data/train-images.idx3-ubyte")
train_y = load_labels("data/train-labels.idx1-ubyte")

test_X = load_images("data/t10k-images.idx3-ubyte")
test_y = load_labels("data/t10k-labels.idx1-ubyte")


# Keep only 0,1,2

def filter_012(X, y):

    idx = (y==0) | (y==1) | (y==2)

    return X[idx], y[idx]


train_X, train_y = filter_012(train_X, train_y)
test_X, test_y = filter_012(test_X, test_y)


# Normalize (0 to 1)


train_X = train_X / 255.0
test_X = test_X / 255.0



# Sample 100 per class


np.random.seed(42)
os.makedirs("results", exist_ok=True)

def sample_data(X, y):

    X_new = []
    y_new = []

    for c in [0,1,2]:

        ind = np.where(y==c)[0]

        choose = np.random.choice(ind, 100, replace=False)

        X_new.append(X[choose])
        y_new.append(y[choose])

    return np.vstack(X_new), np.hstack(y_new)


train_X, train_y = sample_data(train_X, train_y)
test_X, test_y = sample_data(test_X, test_y)



# MLE Mean


def compute_mean(X):

    mu = np.sum(X, axis=0) / X.shape[0]

    return mu



# MLE Covariance


def compute_cov(X, mu):

    N = X.shape[0]
    d = X.shape[1]

    cov = np.zeros((d,d))

    for i in range(N):

        diff = (X[i] - mu).reshape(d,1)

        cov = cov + diff @ diff.T

    cov = cov / N

    return cov



# Compute Parameters

means = {}
covs = {}
priors = {}

for c in [0,1,2]:

    Xc = train_X[train_y==c]

    means[c] = compute_mean(Xc)

    covs[c] = compute_cov(Xc, means[c])

    priors[c] = Xc.shape[0] / train_X.shape[0]



# LDA


# Pooled covariance

d = train_X.shape[1]

Sigma = np.zeros((d,d))

for c in [0,1,2]:

    Sigma = Sigma + priors[c]*covs[c]


# Regularization
Sigma = Sigma + 1e-3*np.eye(d)

Sigma_inv = np.linalg.inv(Sigma)


def lda_g(x, mu, prior):

    val = x @ Sigma_inv @ mu

    val = val - 0.5*(mu @ Sigma_inv @ mu)

    val = val + np.log(prior)

    return val


def lda_predict(X):

    preds = []

    for x in X:

        scores = []

        for c in [0,1,2]:

            scores.append(lda_g(x,means[c],priors[c]))

        preds.append(np.argmax(scores))

    return np.array(preds)


# QDA

def qda_g(x, mu, cov, prior):

    d = len(mu)

    cov = cov + 1e-1*np.eye(d)

    cov_inv = np.linalg.inv(cov)

    sign, logdet = np.linalg.slogdet(cov)

    diff = x - mu

    val = -0.5*logdet

    val = val - 0.5*(diff @ cov_inv @ diff)

    val = val + np.log(prior)

    return val


def qda_predict(X):

    preds = []

    for x in X:

        scores = []

        for c in [0,1,2]:

            scores.append(qda_g(x,means[c],covs[c],priors[c]))

        preds.append(np.argmax(scores))

    return np.array(preds)



# Accuracy


def accuracy(y, yhat):

    return np.mean(y==yhat)


lda_pred = lda_predict(test_X)
qda_pred = qda_predict(test_X)


print("LDA Accuracy:", accuracy(test_y, lda_pred))
print("QDA Accuracy:", accuracy(test_y, qda_pred))



# Discriminant Example


x0 = test_X[0]
y0 = test_y[0]

print("\nTrue Label:", y0)


print("\nLDA Scores:")

for c in [0,1,2]:

    print(c, lda_g(x0,means[c],priors[c]))


print("\nQDA Scores:")

for c in [0,1,2]:

    print(c, qda_g(x0,means[c],covs[c],priors[c]))



# t-SNE Plot


tsne = TSNE(n_components=2, random_state=42)

train_2d = tsne.fit_transform(train_X)
test_2d = tsne.fit_transform(test_X)


plt.figure(figsize=(12,5))


plt.subplot(1,2,1)

plt.title("Train t-SNE")

plt.scatter(train_2d[:,0],train_2d[:,1],
            c=train_y,cmap="jet")

plt.colorbar()


plt.subplot(1,2,2)

plt.title("Test t-SNE")

plt.scatter(test_2d[:,0],test_2d[:,1],
            c=test_y,cmap="jet")

plt.colorbar()
plt.savefig("results/tsne_plots.png", dpi=300, bbox_inches="tight")

plt.show()
