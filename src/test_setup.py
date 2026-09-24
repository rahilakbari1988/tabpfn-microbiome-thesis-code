import torch
import time
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tabpfn import TabPFNClassifier, TabPFNRegressor

print("PyTorch:", torch.__version__)
print("CUDA OK:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

print("--- Classification ---")
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
t0 = time.time()
clf = TabPFNClassifier(device="cuda")
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)
print("Accuracy:", round(accuracy_score(yte, pred), 3))
print("Time:", round(time.time()-t0, 2))

print("--- Regression ---")
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
t0 = time.time()
reg = TabPFNRegressor(device="cuda")
reg.fit(Xtr, ytr)
pred = reg.predict(Xte)
print("R2:", round(r2_score(yte, pred), 3))
print("Time:", round(time.time()-t0, 2))

print("DONE")