"""
line fitting taught us predict->measure error->adjust parameters

now we switch from scores to yes or no, to PROBABILITY
email spam or not
tumor benign or not
transaction fraud or legit

we use logistic regression for this

its basically linear regression followed by a squashing function
squashing function here is sigmoid
it converts any number to a number between range 0 and 1

large negative numbers ->near 0
large positive numbers ->near 1

therefore the model will do linear regression and find a linear value using z=w*x+b, then it will convert to probability using sigmoid
prediction=sigmoid(z)

if probability is above 0.5, it belongs to class 1 else 0
"""

import numpy as np
import matplotlib.pyplot as plt

#dataset
x=np.array([1,2,3,4,5,6])
y=np.array([0,0,0,1,1,1]) # means, 1,2,3 belongs to class 0 and 4,5,6 belongs to class 1

"""
Think of x as an input variable.
example: 
x is hours studied
And y is the correct label.

0 = fail
1 = pass

So the dataset says:

1 hour → fail
2 hours → fail
3 hours → fail
4 hours → pass
5 hours → pass
6 hours → pass

model's job is to discover the boundary between failing and passing"""

w=0.0
b=0.0

learning_rate=0.01
epochs=1000

def sigmoid(z):
    """
    model first calculates the linear value z=w*x+b, but that value could be any number like -100,200 etc, sigmoid compresses it between 0 and 1"""
    return 1/(1+np.exp(-z))


#training begins
for i in range(epochs):

    z=w*x+b
    y_pred=sigmoid(z)
    #now the predictions look like [0.02, 0.05, 0.20, 0.70, 0.90, 0.97]

    error=y_pred-y 
    """If the model predicts 0.7 but the correct label is 1: error = -0.3
    If it predicts 0.9 but the label is 0: error = 0.9
    the sign tells us the direction of the mistake."""


    #compute gradients
    dw=(error*x).mean()
    db=error.mean()

    w=w-learning_rate*dw
    b=b-learning_rate*db


#After training, we print the learned parameters
print("weight: ",w)
print("bias: ",b)


#now we prepare a smooth curve for visualization, linspace generates 100 evenly spaced values between 0 and 7
x_line=np.linspace(0,7,100)

#this produces the probability curve
y_line=sigmoid(w*x_line+b)


plt.scatter(x,y) # this plots the original data points (1,0) (2,0) (3,0), (4,1) (5,1) (6,1)
plt.plot(x_line,y_line) # generates s shaped curve
plt.show()