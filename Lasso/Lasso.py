import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("CE 475 Course Project Data.csv")

a=data.x1.values
b=data.x2.values
c=data.x3.values
d=data.x4.values
e=data.x5.values
f=data.x6.values
g=data.Y.values


x1=np.array(a)
x2=np.array(b)
x3=np.array(c)
x4=np.array(d)
x5=np.array(e)
x6=np.array(f)
ones=np.ones((1,len(x1)))
x=np.vstack((ones,x1,x2,x3,x4,x5,x6))
x=x.T
y=np.array(g)
finx=x[100:]
unky=y[100:]
clx=np.delete(x,range(100,120),0)
cly=np.delete(y,range(100,120),0)


lamb=np.arange(0,100,10)
errorsq=np.array([])
MSE=np.array([])
coef=np.array([])
pred=np.array([])
sqer=0
for i in lamb:
    for j in range(len(clx)):
        train=np.delete(clx,j,0)
        trainy=np.delete(cly,j,0)
        model = skl.Lasso(alpha=i)
        model.fit(train, trainy)

        test=clx[j]
        test=test.reshape(1,-1)
        predlas=model.predict(test)
        pred=np.append(pred,predlas)
        error = cly[j] - predlas
        error =np.square(error)
        sqer += error
    coef = np.append(coef,model.coef_)
    MSE = np.append(MSE, sqer / len(x))
coef=np.reshape(coef,(10,7))

X1 = np.array([])
X2 = np.array([])
X3 = np.array([])
X4 = np.array([])
X5 = np.array([])
X6 = np.array([])


for i in coef:
    X1 = np.append(X1, i[1])
    X2 = np.append(X2, i[2])
    X3 = np.append(X3, i[3])
    X4 = np.append(X4, i[4])
    X5 = np.append(X5, i[5])
    X6 = np.append(X6, i[6])

plt.figure()
plt.plot(lamb, X1, c='k',label="x1")
plt.plot(lamb, X2, c='y',label="x2")
plt.plot(lamb, X3, c='b',label="x3")
plt.plot(lamb, X4, c='c',label="x4")
plt.plot(lamb, X5, c='r',label="x5")
plt.plot(lamb, X6, c='g',label="x6")
plt.legend()
plt.show()
