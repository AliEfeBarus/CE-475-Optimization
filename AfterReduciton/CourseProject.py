import numpy as np
import pandas as pd


data=pd.read_csv("CE 475 Course Project Data.csv")

a=data.x3.values
b=data.x5.values
c=data.x6.values
d=data.Y.values


x3=np.array(a)
x5=np.array(b)
x6=np.array(c)

ones=np.ones((1,len(x3)))
x=np.vstack((ones,x3,x5,x6))
x=x.T
y=np.array(d)
finx=x[100:]
unky=y[100:]
clx=np.delete(x,range(100,120),0)
cly=np.delete(y,range(100,120),0)
def ad_r2(y,yest,d):
    yavg = np.mean(y)
    rss = 0
    tss = 0
    for i in range(len(y)):
        rss += (y[i] - yest[i]) ** 2
        tss += (y[i] - yavg) ** 2

    ad_r2 = 1 - ((rss / (len(y) - d - 1)) / (tss / (len(y) - 1)))
    return ad_r2
def r_squared(yact,yest):
    yavg=np.mean(yact)
    rss=0
    tss=0
    for i in range(len(yact)):
        rss+=(yact[i]-yest[i])**2
        tss+=(yact[i]-yavg)**2

    rsqr=1-(rss/tss)

    return rsqr
def mullin_coef(x,y):
    biden=np.dot(x.T,x)
    biden=np.linalg.pinv(biden)
    b=np.dot(biden,np.dot(x.T,y))
    return b
def mse_calc(y, ypred):
    mse_num = 0
    for i in range(len(y)):
        mse_num += np.square(y[i] - ypred[i])
    return mse_num / len(y)

mse=np.array([])
k =5
print('Folds: ', k)


for i in range(0, len(clx), int(len(clx) / k)):
    x_test = clx[i:i + int(len(clx) / k)]
    y_test = cly[i:i + int(len(clx) / k)]
    x_train = np.delete(clx, range(i, i + int(len(clx) / k)), 0)
    y_train = np.delete(cly, range(i, i + int(len(clx) / k)), 0)

    beta = mullin_coef(x_train, y_train)
    ypred1 = np.dot(x_test, beta)
    for i in range(len(ypred1)):
        if ypred1[i] < 0:
            ypred1[i] = 0

    m = mse_calc(ypred1, y_test)
    mse = np.append(mse, m)

a = np.min(mse)

for i in range(len(mse)):
    if mse[i] == a:
        print(i)
        index = i

index = (index * int(len(clx) / k))
x_test = clx[index:index + int(len(clx) / k)]
y_test = cly[index:index+ int(len(clx) / k)]

x_train = np.delete(clx, range(index, index + int(len(clx) / k)), 0)
y_train = np.delete(cly, range(index, index + int(len(clx) / k)), 0)


beta = mullin_coef(x_train, y_train)
ypred2 = np.dot(x_test, beta)
for i in range(len(ypred2)):
    if ypred2[i] < 0:
        ypred2[i] = 0
ypred3 = np.dot(finx, beta)

rsq = r_squared(y_test, ypred2)
adjr = ad_r2(y_test, ypred2, 3)
print('RSQ:', rsq)
print()
print('Adjusted R Square:', adjr)
print()
print(ypred3)