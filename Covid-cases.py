import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg 
import math

from statsmodels.tsa.base import prediction

# reading the data
data = pd.read_csv('daily_covid_cases.csv', delimiter= ',')


plt.figure(figsize= (12,14))
plt.plot(data['Date'] , data['new_cases'])
xticks = ['Feb-20','Apr-20','Jun-20','Aug-20','Oct-20','Dec-20','Feb-21','Apr-21','Jun-21','Aug-21','Oct-21']
plt.xticks([i for i in range(int(612/11),612,int(612/11)) ], xticks, rotation = 45)
plt.xlabel("Month-Year")
plt.ylabel("New Confirmed Cases")
plt.show()

# making a new data frame for 1 day lag
data_1 = pd.DataFrame()  
data_1 = data.iloc[1:]
data_x  = data.iloc[:611]

# calculating the pearson correlation using scipy library
print("the pearson correlation between the original data and a one day lagged data is: " ,pearsonr(list(data_x['new_cases']), list(data_1['new_cases']))[0])


plt.scatter(data_x['new_cases'], data_1['new_cases'])
plt.xlabel("actual cases")
plt.ylabel("one day lagged cases")
plt.show()

# declaring a empty list to append the values of correlation 
corr_lst = []
lag_lst = [1,2,3,4,5,6]
for i in lag_lst:
# making a new data frame for lagged data
    data_i = pd.DataFrame()
    data_i = data.iloc[i:]
    data_x  = data.iloc[:612-i]
# appending the values of correlation for different lags
    corr_lst.append(pearsonr(list(data_x['new_cases']), list(data_i['new_cases']))[0])
# plotting the line plot of all the correlation values
plt.plot(lag_lst , corr_lst)
plt.xlabel("no. of lags")
plt.ylabel("correlation coefficient")
plt.show()



# plotting the correlation coefficient using plot_acf
plot_acf(data['new_cases'] , lags = 6)
plt.grid()
plt.yticks(np.arange(-1.25,1.5,step = 0.25))
plt.xlabel("no. of lags")
plt.ylabel("correlation coefficient by plot_acf function")
plt.show()


# reading csv and splitting the data with test size=0.35
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train_, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]
# converting the series of arrays into a list of numbers 
train = []
for i in range (len(train_)):
    train.append(train_[i])


# building the autp regression model using the code from the snippet given in the question pdf
Window = 5 # The lag=5
model = AutoReg(train, lags=Window) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print("coefficients w0,w1,w2,w3,w4,w5 respictively are:",coef)

#designing a auto regression model and predicting the test values
history = train[len(train)-Window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-Window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(Window):
        yhat += coef[d+1] * lag[Window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.



# plotting the scatter plot of the predicted value and the actual value
plt.scatter(test,predictions)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.show()

# plotting the line plot of the predicted values and the actual values
plt.plot(predictions )
plt.plot(test)
xticks = ['Apr-21','Jun-21','Aug-21','Oct-21']
plt.xticks([i for i in range(int(215/4),215,int(215/4)) ], xticks, rotation = 45)
plt.xlabel("Month-Year")
plt.ylabel("no. of cases")
plt.show()

n=len(test)
s=0

# calculating the rmse(%)
for i in range(n):
    s=s+(predictions[i]-test[i])**2
avg=sum(test)/len(test)
rmse=(math.sqrt(s/len(test))/avg)*100
print("rmse is",rmse)

# calculating the MAPE :

s=0
for i in range(n):
    s=s+ abs(predictions[i]-test[i])/test[i]
mape=(s/n)*100
print("MApe is:",mape)



# defining a function to calculate rmse(%) and MAPE using the input value for lags
def auto_reg(p):
        Window = p # The lag=p
        model = AutoReg(train, lags=Window) 
        model_fit = model.fit() # fit/train the model
        coef = model_fit.params # Get the coefficients of AR model
        history = train[len(train)-Window:]
        history = [history[i] for i in range(len(history))]
        predictions = list() # List to hold the predictions, 1 step at a time
        for t in range(len(test)):
            length = len(history)
            lag = [history[i] for i in range(length-Window,length)]
            yhat = coef[0] # Initialize to w0
            for d in range(Window):
                yhat += coef[d+1] * lag[Window-d-1] # Add other values
            obs = test[t]
            predictions.append(yhat) #Append predictions to compute RMSE later
            history.append(obs) # Append actual test value to history, to be used in next step.

        # calculating the rmse(%)

        n=len(test)
        s=0
        for i in range(n):
            s=s+(predictions[i]-test[i])**2
        avg=sum(test)/len(test)
        rmse=(math.sqrt(s/len(test))/avg)*100

        # calculating the MAPE

        s=0
        for i in range(n):
            s=s+ abs(predictions[i]-test[i])/test[i]
        mape=(s/n)*100

        return rmse[0],mape[0]
# declaring the lists for rmse and mape       
rmse=[]
mape=[]
p = []
# getting the values of rmse and MAPE using the above defined function
for i in [0,1,2,3,5]:
    if i==0 :
        a , b = auto_reg(1)
        rmse.append(a)
        mape.append(b)
        p.append(1)
    else :
        a , b = auto_reg(5*i)
        rmse.append(a)
        mape.append(b)
        p.append(5*i)
for i in range (len(rmse)):
    print("rmse for lag =",p[i],"is:", rmse[i])
    print("MAPE for lag =",p[i],"is:", mape[i])
# plotting the bar graph of rmse
plt.bar(p,rmse)
plt.xticks(p)
plt.xlabel("on. of lags")
plt.ylabel("RMSE(%)")
plt.show()
# plotting the bar grapf of MAPE
plt.bar(p,mape)
plt.xticks(p)
plt.xlabel("no. of lags")
plt.ylabel("MAPE")
plt.show()



# computing the optimal no. of lags, which also satisfies the givrn condition 
p=1
flag=1
while(flag==1):
    new_train=train[p:]
    l=len(new_train)
    lag_new_train=train[:l]
    # changing the dimensions of series'
    nt =[]
    lnt =[]
    for i in range (len(new_train)):
        nt.append(new_train[i][0])
        lnt. append(lag_new_train[i][0])
    corr = pearsonr(lnt,nt)
    if(2/math.sqrt(l)>abs(corr[0])):
        flag=0
    else:
        p=p+1

print("optimal no. of lags:",p-1)

# building a auto regression model using the code from thr snippet and optimal lag value
Window = p-1
model = AutoReg(train, lags=Window) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
history = train[len(train)-Window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-Window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(Window):
        yhat += coef[d+1] * lag[Window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

# calculating the rmse for the above prediction
n=len(test)
s=0
for i in range(n):
    s=s+(predictions[i]-test[i])**2
avg=sum(test)/len(test)
rmse=(math.sqrt(s/len(test))/avg)*100
print("rmse is " ,rmse)

# Calculating MAPE
s=0
for i in range(n):
    s=s+ abs(predictions[i]-test[i])/test[i]
mape=(s/n)*100
print("mape is:",mape)
