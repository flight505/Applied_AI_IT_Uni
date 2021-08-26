def predict(theta, X, y, alpha, epochss):
    J, th = gradient_descent(theta, X, y, alpha, epochs)
    h = hypothesis(X, theta)
    for i in range(len(h)):
        h[i]=1 if h[i]>=0.5 else 0
    y = list(y)
    acc = np.sum([y[i] == h[i] for i in range(len(y))])/len(y)
    return J, acc