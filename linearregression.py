"""
hours studied  ->exam score
2->40
3->50
4->60

the machine must discover the pattern, we will use linear regression 
it uses y=wx+b, y is the predicted output score, x is the input hours, w is the weight or slope of line, b is the bias where the line crosses the axis 

job of this model is to simply learn the correct values of weight and bias 
..
"""



import numpy as np #allows arrays and fastmath

# training data
x = np.array([2,3,4,5], dtype=float) #creates input data
y = np.array([40,50,60,70], dtype=float) #output data

# at first the model knows nothing, so it predicts a flat line so the w and b are 0 
w = 0.0
b = 0.0

# then we define the learning speed, this controls how big steps the model take towards adjusting itself, too big ->jumps wildly, too small ->very slow
learning_rate = 0.01
epochs = 1000 # epoch means how many times the model sees the entire data set, 1000 times means it will practice 1000 times

for i in range(epochs): #learning loop

    y_pred = w*x + b
    """ if the model currently believes w=5 and b=10 then for x=2,, y_pred=5*2+10, which may or may not be wrong, but the learning begins with wrong guesses"""

    error = y_pred - y 
    """
    this error will tell us how far the prediction is from reality,
      suppose the actual value is 40 but prediction is 20, error is 20-40=-20, negative sign is important and it tells us that prediction is too small, if the error was positive, the predictino would have been too big,

      So error tells us two things:

• how wrong the model is
• which direction the mistake goes

in our dataset, we calculate errors for all points at once because numpy works with arrays ie
y_pred = [20, 25, 30, 35]
y      = [40, 50, 60, 70]

and thus the error becomes

error = [-20, -25, -30, -35]
    """
     

    loss = (error**2).mean()

    """
    error**2 squares every error,
    Example [-20, -25, -30, -35] becomes [400, 625, 900, 1225]
    squaring is done because
        Negative errors shouldn't cancel positive ones while we are finding mean
        Bigger mistakes get punished more
    now we average(mean) them
    (400 + 625 + 900 + 1225) / 4 -> this is the loss

    loss -> basically how bad the model is overall, if the model improves the number goes down
    """

    """
    now the model needs to answer, should i increase or decrease w, if yes then, by how much, below 2 lines are called gradients
    
    gradient is basically, how sensitive the error is to a parameter"""
    dw = (2*(error*x)).mean() #dw is basically, how the error changes if w changes,
    #now error is multiplied with x here, why? in our eqn y=wx+b, there w is multiplied with b, since w influences prediction through x, the gradient also depends on x
    db = (2*error).mean()#db is, how the error changes is b changes

    #update the parameters
    """
    example:
    dw = -100
    learning_rate = 0.01
    w = w - (0.01 * -100) -> w=w+1
    why?
    because the gradient has told us, that by increasing w, error will be reduced
    """
    w = w - learning_rate*dw
    b = b - learning_rate*db

    """after repeating this process many times, every loop will 
            make predictions
            measure error
            compute gradient
            adjust parameters

            now we are having 2 parameters, in real world, we would have 10 billion parameters and massive datasets
    """

print("weight:", w)
print("bias:", b)