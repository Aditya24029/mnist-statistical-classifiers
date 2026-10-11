import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs("results", exist_ok=True)


# Load MNIST IDX files


def load_images(file):
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    images = data[16:].reshape(-1, 784)
    return images / 255.0

def load_labels(file):
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    labels = data[8:]
    return labels


print("Loading MNIST...")

X_train_full = load_images("data/train-images.idx3-ubyte")
y_train_full = load_labels("data/train-labels.idx1-ubyte")

X_test_full = load_images("data/t10k-images.idx3-ubyte")
y_test_full = load_labels("data/t10k-labels.idx1-ubyte")



# Keep only digits 0,1,2


mask_train = np.isin(y_train_full, [0,1,2])
mask_test = np.isin(y_test_full, [0,1,2])

X_train_full = X_train_full[mask_train]
y_train_full = y_train_full[mask_train]

X_test_full = X_test_full[mask_test]
y_test_full = y_test_full[mask_test]



# Select 100 samples per class


trainX=[]
trainY=[]
testX=[]
testY=[]

for c in [0,1,2]:

    idx_train = np.where(y_train_full==c)[0]
    idx_test = np.where(y_test_full==c)[0]

    np.random.shuffle(idx_train)
    np.random.shuffle(idx_test)

    trainX.append(X_train_full[idx_train[:100]])
    trainY.append(y_train_full[idx_train[:100]])

    testX.append(X_test_full[idx_test[:100]])
    testY.append(y_test_full[idx_test[:100]])

X_train = np.vstack(trainX).T
y_train = np.hstack(trainY)

X_test = np.vstack(testX).T
y_test = np.hstack(testY)



# PCA


def PCA(X, variance):

    mu = np.mean(X, axis=1, keepdims=True)
    Xc = X - mu

    S = (Xc @ Xc.T) / (X.shape[1] - 1)

    eigvals, eigvecs = np.linalg.eigh(S)

    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    cumulative = np.cumsum(eigvals) / np.sum(eigvals)

    k = np.where(cumulative >= variance)[0][0] + 1

    Up = eigvecs[:,:k]

    Y = Up.T @ Xc

    return Y, Up, mu, k, cumulative



# Reconstruction


def reconstruct(Y, Up, mu):
    return Up @ Y + mu



# Accuracy


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)



# LDA


class LDA:

    def fit(self, X, y):

        self.classes = np.unique(y)
        self.means = {}
        self.priors = {}

        n = X.shape[1]
        d = X.shape[0]

        Sw = np.zeros((d,d))

        for c in self.classes:

            Xc = X[:,y==c]
            mu = np.mean(Xc, axis=1, keepdims=True)

            self.means[c] = mu
            self.priors[c] = Xc.shape[1]/n

            diff = Xc - mu
            Sw += diff @ diff.T

        self.Sigma = Sw/(n-len(self.classes))
        self.Sigma_inv = np.linalg.inv(self.Sigma)

    def predict(self, X):

        preds=[]

        for i in range(X.shape[1]):

            x = X[:,i:i+1]

            scores=[]

            for c in self.classes:

                mu = self.means[c]

                g = (x.T@self.Sigma_inv@mu
                     -0.5*mu.T@self.Sigma_inv@mu
                     +np.log(self.priors[c]))

                scores.append(g.item())

            preds.append(self.classes[np.argmax(scores)])

        return np.array(preds)



# QDA


class QDA:

    def fit(self, X, y):

        self.classes = np.unique(y)
        self.means = {}
        self.covs = {}
        self.priors = {}

        for c in self.classes:

            Xc = X[:,y==c]

            mu = np.mean(Xc, axis=1, keepdims=True)

            self.means[c] = mu
            self.priors[c] = Xc.shape[1]/X.shape[1]

            diff = Xc - mu
            cov = (diff @ diff.T)/(Xc.shape[1]-1)

            cov += 1e-6*np.eye(cov.shape[0])

            self.covs[c] = cov

    def predict(self, X):

        preds=[]

        for i in range(X.shape[1]):

            x = X[:,i:i+1]

            scores=[]

            for c in self.classes:

                mu = self.means[c]
                cov = self.covs[c]

                inv = np.linalg.inv(cov)
                det = np.linalg.det(cov)

                diff = x - mu

                g = (-0.5*np.log(det)
                     -0.5*(diff.T@inv@diff)
                     +np.log(self.priors[c]))

                scores.append(g.item())

            preds.append(self.classes[np.argmax(scores)])

        return np.array(preds)



# PCA 75%


Y75,Up75,mu75,k75,cumulative = PCA(X_train,0.75)
Xtest75 = Up75.T @ (X_test - mu75)

print("PCA components (75% variance):",k75)

# Plot explained variance curve 
plt.figure()
plt.plot(cumulative)
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance curve")
plt.grid(True)
plt.savefig("results/pca_variance_curve.png", dpi=300)


# Reconstruction visualization

print("\nReconstruction MSE:")

plt.figure(figsize=(12,4))

for i in range(5):

    original = X_train[:,i].reshape(28,28)
    recon = reconstruct(Y75[:,i:i+1],Up75,mu75).reshape(28,28)

    mse = np.mean((X_train[:,i]-recon.flatten())**2)

    print("Sample",i,":",mse)

    plt.subplot(2,5,i+1)
    plt.imshow(original,cmap="gray")
    plt.axis("off")

    plt.subplot(2,5,i+6)
    plt.imshow(recon,cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.savefig("results/pca_reconstruction.png", dpi=300)


# PCA visualization

plt.figure()

for c in [0,1,2]:
    idx = y_train==c
    plt.scatter(Y75[0,idx],Y75[1,idx],label=f"Digit {c}")

plt.legend()
plt.title("PCA Projection")
plt.savefig("results/pca_projection.png", dpi=300)



# FDA


def FDA(Y,y):

    classes = np.unique(y)

    mu = np.mean(Y,axis=1,keepdims=True)

    Sw = np.zeros((Y.shape[0],Y.shape[0]))
    Sb = np.zeros((Y.shape[0],Y.shape[0]))

    for c in classes:

        Xc = Y[:,y==c]

        muc = np.mean(Xc,axis=1,keepdims=True)

        diff = Xc-muc
        Sw += diff@diff.T

        Nc = Xc.shape[1]

        mdiff = muc-mu
        Sb += Nc*(mdiff@mdiff.T)

    eigvals,eigvecs = np.linalg.eig(np.linalg.pinv(Sw)@Sb)

    idx = np.argsort(eigvals)[::-1]

    W = np.real(eigvecs[:,idx[:2]])

    return W


W = FDA(Y75,y_train)

Z_train = W.T @ Y75
Z_test = W.T @ Xtest75

plt.figure()

for c in [0,1,2]:

    idx = y_train==c
    plt.scatter(Z_train[0,idx],Z_train[1,idx],label=f"Digit {c}")

plt.legend()
plt.title("FDA Projection")
plt.savefig("results/fda_projection.png", dpi=300)



# FDA Classification


lda_fda = LDA()
qda_fda = QDA()

lda_fda.fit(Z_train,y_train)
qda_fda.fit(Z_train,y_train)

print("\nFDA Results")

print("LDA Train:",accuracy(y_train,lda_fda.predict(Z_train)))
print("LDA Test:",accuracy(y_test,lda_fda.predict(Z_test)))

print("QDA Train:",accuracy(y_train,qda_fda.predict(Z_train)))
print("QDA Test:",accuracy(y_test,qda_fda.predict(Z_test)))



# PCA 75% Classification


lda = LDA()
qda = QDA()

lda.fit(Y75,y_train)
qda.fit(Y75,y_train)

print("\nPCA 75%")

print("LDA Train:",accuracy(y_train,lda.predict(Y75)))
print("LDA Test:",accuracy(y_test,lda.predict(Xtest75)))

print("QDA Train:",accuracy(y_train,qda.predict(Y75)))
print("QDA Test:",accuracy(y_test,qda.predict(Xtest75)))



# PCA 90%


Y90,Up90,mu90,k90,_ = PCA(X_train,0.90)
Xtest90 = Up90.T@(X_test-mu90)

lda90=LDA()
qda90=QDA()

lda90.fit(Y90,y_train)
qda90.fit(Y90,y_train)

print("\nPCA 90%")

print("LDA Train:",accuracy(y_train,lda90.predict(Y90)))
print("LDA Test:",accuracy(y_test,lda90.predict(Xtest90)))

print("QDA Train:",accuracy(y_train,qda90.predict(Y90)))
print("QDA Test:",accuracy(y_test,qda90.predict(Xtest90)))



# First 2 PCs


Up2 = Up75[:,:2]

Y2 = Up2.T@(X_train-mu75)
Xtest2 = Up2.T@(X_test-mu75)

lda2=LDA()
qda2=QDA()

lda2.fit(Y2,y_train)
qda2.fit(Y2,y_train)

print("\nUsing only first 2 PCs")

print("LDA Train:",accuracy(y_train,lda2.predict(Y2)))
print("LDA Test:",accuracy(y_test,lda2.predict(Xtest2)))

print("QDA Train:",accuracy(y_train,qda2.predict(Y2)))
print("QDA Test:",accuracy(y_test,qda2.predict(Xtest2)))


plt.show()
