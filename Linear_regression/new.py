import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("univariate_regression_data.csv")
# Creating w and b and x
X = data["Feature"].values
Y = data["Target"].values
w = 0
b = 0
m = X.shape[0]

plt.scatter(data["Feature"],data["Target"],c='red',marker="X")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression")
plt.show()

def cost_func(w,X,Y,b,m):
    sumerr = 0
    for i in range(m):
        t_err = ((w*X[i]+b)- Y[i])**2
        sumerr = sumerr + t_err
    mse = (sumerr)/(2*m)
    return mse

def Der_gradient(w,b,X,Y,m):
#This function is going to find the derivative for the gradient descent
    dj_db = 0
    dj_dw = 0
    for i in range(m):
        t1_err = ((w*X[i]+b)- Y[i])
        t2_err = ((w*X[i]+b)- Y[i])*X[i]
        dj_db  = dj_db + t1_err
        dj_dw  = dj_dw + t2_err
        
    dj_db /= m
    dj_dw /= m
    return dj_db,dj_dw
        
    
def gradient_descent(w,b,X,Y,n_iteration,alpha):
    m = X.shape[0]
    for i in range(n_iteration):
        dj_db , dj_dw = Der_gradient(w,b,X,Y,m)
        w = w - (alpha*dj_dw)
        b = b - (alpha*dj_db)
        if i % 100 == 0:
            mse = cost_func(w,X,Y,b,m)
            print(f"COST {mse}")
    return w,b
    
    
def predict(w, x_in, b, X, Y):
    y_hat = w * x_in + b
    return y_hat

w, b = gradient_descent(w, b, X, Y, 10000, 0.001)
x_line = np.linspace(X.min(),X.max(),100)
y_line = w * x_line + b
plt.scatter(data["Feature"],data["Target"],c='red',marker="X")
plt.plot(x_line,y_line,c='blue')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression")
plt.show()

x_in = int(input("Enter the x value to predict for: "))
val = predict(w,x_in,b,X,Y)


print(f"The predicted output is {val}")