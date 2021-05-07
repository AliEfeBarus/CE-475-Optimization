import numpy as num
import csv
import matplotlib.pyplot as plt
import sklearn.linear_model as ln
import warnings

warnings.filterwarnings("ignore")

X1 = num.array([])
X2 = num.array([])
X3 = num.array([])
X4 = num.array([])
X5 = num.array([])
X6 = num.array([])
Y = num.array([])
errorsq = num.array([])
MSE = num.array([])
coeffs = num.array([])
pred = num.array([])
sqer = 0

with open("CE 475 Course Project Data.csv") as f:
    mylist = list(csv.reader(f))

for row in mylist:
    if row != mylist[0]:
        X1 = num.append(X1, int(row[1]))
        X2 = num.append(X2, int(row[2]))
        X3 = num.append(X3, int(row[3]))
        X4 = num.append(X4, int(row[4]))
        X5 = num.append(X5, int(row[5]))
        X6 = num.append(X6, int(row[6]))
        if row[7] == '':
            continue
        Y = num.append(Y, int(row[7]))
ones = num.ones((1, len(X1)))
x = num.vstack((ones, X1, X2, X3, X4, X5, X6))
x = x.T

mustpredx = x[100:]
xcut = num.delete(x, range(100, 120), 0)
delta = num.arange(0, 1000, 10)

for index in delta:
    for xin in range(len(xcut)):
        X_train = num.delete(xcut, xin, axis=0)
        Y_train = num.delete(Y, xin, axis=0)
        lasso = ln.Lasso(alpha=index)
        lasso.fit(X_train, Y_train)

        X_test = xcut[xin]
        X_test = X_test.reshape(1, -1)
        lp = lasso.predict(X_test)
        pred = num.append(pred, lp)

        error = Y[xin] - lp
        error = num.square(error)
        sqer += error
    coeffs = num.append(coeffs, lasso.coef_)
    MSE = num.append(MSE, sqer / len(x))
    sqer = 0

coeffs = num.reshape(coeffs, (100, 7))

x1 = num.array([])
x2 = num.array([])
x3 = num.array([])
x4 = num.array([])
x5 = num.array([])
x6 = num.array([])

for row in coeffs:
    x1 = num.append(x1, row[1])
    x2 = num.append(x2, row[2])
    x3 = num.append(x3, row[3])
    x4 = num.append(x4, row[4])
    x5 = num.append(x5, row[5])
    x6 = num.append(x6, row[6])

plt.figure()
plt.plot(delta, x1, c='b')
plt.plot(delta, x2, c='g')
plt.plot(delta, x3, c='r')
plt.plot(delta, x4, c='c')
plt.plot(delta, x5, c='k')
plt.plot(delta, x6, c='y')
plt.show()
