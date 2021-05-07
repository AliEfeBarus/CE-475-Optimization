import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv("CE 475 Course Project Data.csv")

a=data.x1.values
b=data.x2.values
c=data.x3.values
d=data.x4.values
e=data.x5.values
f=data.x6.values
g=data.Y.values

rsq1=np.array([])
rsq2=np.array([])

adr1=np.array([])
adr2=np.array([])
preds=np.array([])


x1=np.array(a)
x2=np.array(b)
x3=np.array(c)
x4=np.array(d)
x5=np.array(e)
x6=np.array(f)
x=np.vstack((x1,x2,x3,x4,x5,x6))
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

for i in range(1,201):
    rf1=RandomForestRegressor(max_depth=6,n_estimators=i)
    rf2=RandomForestRegressor(max_depth=6,n_estimators=i,max_features="sqrt")

    rf1.fit(clx, cly)
    ypred1 = rf1.predict(clx)
    predd = rf1.predict(finx)
    preds = np.append(preds, predd, -1)
    rsq1 = np.append(rsq1, r_squared(cly, ypred1))
    adr1 = np.append(adr1, ad_r2(cly, ypred1, 6))

    rf2.fit(clx, cly)
    pred2 = rf2.predict(clx)
    rsq2 = np.append(rsq2, r_squared(cly, pred2))
    adr2 = np.append(adr2, ad_r2(cly, pred2, 6))


preds= np.reshape(preds, (200, 20))


max = np.max(rsq1)
max2= np.max(adr1)
max3 = np.max(rsq2)
max4= np.max(adr2)

print("AUTO-FEATURE SCORES: ")
print("R^2 Score: ", max)
print("Adjusted R^2: ", max2)
print()
print("SQRT-FEATURE SCORES: ")
print("R^2 Score: ", max3)
print("Adjusted R^2: ", max4)
print()
for i in range(len(adr1)):
    if adr1[i] == max2:
        print(preds[i])
