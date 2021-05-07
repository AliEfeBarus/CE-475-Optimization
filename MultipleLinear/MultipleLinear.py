import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
beta=mullin_coef(clx,cly)
clypred=np.dot(clx,beta)
ypred2=np.array([])
for row in finx:
    yest2=(beta[0]*row[0])+(beta[1]*row[1])+(beta[2]*row[2])+(beta[3]*row[3])+(beta[4]*row[4])+(beta[5]*row[5])+(beta[6]*row[6])
    ypred2=np.append(ypred2,yest2)
print("R^2 Score:",r_squared(cly,clypred))
print()
print("Adjusted R^2: ",ad_r2(cly,clypred,6))
print()
print(ypred2)
