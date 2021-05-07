import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv("CE 475 Course Project Data.csv")

a=data.x3.values
b=data.x5.values
c=data.x6.values
d=data.Y.values

mse=np.array([])
rsq1=np.array([])
rsq2=np.array([])
adr1=np.array([])
adr2=np.array([])
preds=np.array([])


x3=np.array(a)
x5=np.array(b)
x6=np.array(c)
x=np.vstack((x3,x5,x6,x3*x5,x3*x6,x5*x6,x3*x3,x5*x5,x6*x6,x3*(x5*x6)))
x=x.T
y=np.array(d)
finx=x[100:]
unky=y[100:]
clx=np.delete(x,range(100,120),0)
cly=np.delete(y,range(100,120),0)
test=clx[60:80]
train=np.delete(clx,range(60,80),0)
trainy=np.delete(cly,range(60,80),0)
testy=cly[60:80]
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

for i in range(1,201):
    rf1=RandomForestRegressor(max_depth=6 ,n_estimators=i)
    rf2=RandomForestRegressor(max_depth=6,n_estimators=i,max_features="sqrt")

    rf1.fit(train, trainy)
    ypred1 = rf1.predict(test)
    predd = rf1.predict(finx)
    preds = np.append(preds, predd, -1)
    rsq1 = np.append(rsq1, r_squared(testy, ypred1))
    adr1 = np.append(adr1, ad_r2(testy, ypred1, 10))

    rf2.fit(train, trainy)
    pred2 = rf2.predict(test)
    rsq2 = np.append(rsq2, r_squared(testy, pred2))
    adr2 = np.append(adr2, ad_r2(testy, pred2, 10))


preds= np.reshape(preds, (200, 20))


print("AUTO-FEATURE SCORES: ")
print("R^2 Score: ", np.max(rsq1))
print("Adjusted R^2: ", np.max(adr1))
print()
print("SQRT-FEATURE SCORES")
print("R^2  Score: ", np.max(rsq2))
print("Adjusted R^2: ", np.max(adr2))


max = np.max(adr1)
for i in range(len(adr1)):
    if adr1[i] == max:
        print(preds[i])