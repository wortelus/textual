import numpy as np
import pickle


def frobenius_loss(X, W, Y, n):
    # ||X W^T - Y||_F^2""
    X_WT = X.dot(W.T)
    diff = X_WT - Y
    return (1 / n) * np.sum(diff ** 2)


def compute_gradient(X, W, Y, n):
    # L = ||X W^T - Y||^2
    X_WT_m_Y = X.dot(W.T) - Y  # [n, dim]
    diff_t = X_WT_m_Y.T  # [dim, n]
    # dL/dW = 2 ( (X W^T - Y)^T X )
    grad = (2 / n) * diff_t.dot(X)  # [dim, dim]
    return grad


def fit(X, Y, alpha=1e-4, steps=1000, tol=1e-6, verbose=True, W=None):
    # počet vzorků
    n = X.shape[0]
    # počet dimenzí
    dims = X.shape[1]

    # Inicializace maticí identity
    if W is None:
        W = np.eye(dims, dtype=np.float32)
    else:
        # Kontrola rozměrů
        if W.shape[0] != dims or W.shape[1] != dims:
            raise ValueError(f"Rozměry W ({W.shape}) neodpovídají rozměrům X ({X.shape})")

    loss = frobenius_loss(X, W, Y, n)
    history = [loss]
    print("Ztráta na začátku:", loss)

    for i in range(steps):
        # výpočet gradientu
        grad = compute_gradient(X, W, Y, n)

        # gradientní krok
        W = W - alpha * grad
        loss = frobenius_loss(X, W, Y, n)
        history.append(loss)

        if verbose and i % 10 == 0:
            print(f"Iter {i}: Loss = {loss:.4f}")

        # Kontrola konvergence
        if len(history) > 10 and abs(history[-10] - history[-1]) < tol:
            if verbose: print(f"Konvergováno po {i} iteracích")
            break
    return W, history


if __name__ == "__main__":
    X_train_path = 'X_train.pkl'
    Y_train_path = 'Y_train.pkl'

    X_train = pickle.load(open(X_train_path, 'rb'))
    Y_train = pickle.load(open(Y_train_path, 'rb'))

    alpha = 1.
    steps = 1500

    print(f"Trénuji transformaci, a={alpha}, steps={steps}")
    W, hist = fit(X_train, Y_train, alpha=alpha, steps=steps, verbose=True)
    with open('W.pkl', 'wb') as f:
        pickle.dump(W, f)

    alpha = 0.1
    steps = 1000
    print(f"Trénuji transformaci, a={alpha}, steps={steps}")
    W, hist_new = fit(X_train, Y_train, alpha=alpha, steps=steps, verbose=True, W=W)
    hist += hist_new


    # plot hist loss
    import matplotlib.pyplot as plt
    plt.plot(hist)
    plt.title("Loss")
    plt.xlabel("Iterace")
    plt.ylabel("Ztráta")
    plt.grid()
    plt.savefig("loss.png")
    plt.show()

    print("done")
