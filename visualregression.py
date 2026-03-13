import matplotlib
matplotlib.use("TkAgg")

import numpy as np 
import matplotlib.pyplot as plt # pythons graphing engine


#training data
x=np.array([2,3,4,5],dtype=float)
y=np.array([40,50,60,70],dtype=float)

#parameters
w=0.0
b=0.0

learning_rate=0.01
epochs=100

#plt.scatter(x,y)  #draws actual data points

plt.ion() #interactive mode

fig,ax=plt.subplots()
ax.scatter(x,y)

line,=ax.plot(x,w*x+b) #creates the line object, instead of redrawing everything, we just update this line

for i in range(epochs):
    y_pred=x*w+b

    error=y_pred-y
    
    dw=(2*(error*x)).mean()
    db=(2*error).mean()

    w=w-learning_rate*dw
    b=b-learning_rate*db

    """if i%10==0: #every 10 iterations, draw the current prediction line
        plt.plot(x,y_pred)
    """


    line.set_ydata(x*w+b)#line visually updates the model, so every time parameter changes, line moves

    plt.draw()
    plt.pause(0.1) #this shows animations

plt.ioff()
plt.show()