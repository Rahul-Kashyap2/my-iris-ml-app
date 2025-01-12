import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_save_models():
    iris = load_iris()
    X, y = iris.data, iris.target  # y in {0,1,2}

    # Model 1: Setosa (1) vs Not Setosa (0)
    y_bin1 = np.where(y == 0, 1, 0)  # 1 if setosa
    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X, y_bin1, test_size=0.2, random_state=42
    )
    model_bin1 = LogisticRegression(max_iter=200)
    model_bin1.fit(X_train1, y_train1)
    acc_bin1 = accuracy_score(y_test1, model_bin1.predict(X_test1))
    print(f"Binary Model1 (Setosa vs Not): {acc_bin1*100:.2f}%")

    joblib.dump(model_bin1, 'models/model_binary1.pkl')
    print("Saved model_binary1.pkl")

    # Model 2: Versicolor(0) vs Virginica(1)
    mask_vv = (y >= 1)
    X_vv = X[mask_vv]
    y_vv = y[mask_vv]
    y_vv = np.where(y_vv == 1, 0, 1)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_vv, y_vv, test_size=0.2, random_state=42
    )
    model_bin2 = LogisticRegression(max_iter=200)
    model_bin2.fit(X_train2, y_train2)
    acc_bin2 = accuracy_score(y_test2, model_bin2.predict(X_test2))
    print(f"Binary Model2 (Versicolor vs Virginica): {acc_bin2*100:.2f}%")

    joblib.dump(model_bin2, 'models/model_binary2.pkl')
    print("Saved model_binary2.pkl")

if __name__ == "__main__":
    train_and_save_models()
