---
layout: post
title: Section 1, PyTorch - ML, Averaged Perceptron
author: Alessandro Santospirito
---

## PyTorch - Basics
## Averaged Perceptron
<img src="../../../../images/1_pytorch-ml/iris-flowers.png" alt="example flowers of dataset" style="width: 45%;"/>

So, the data of the **first** 5 examples looks as follows:

| exmaple# | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | target name|
| --- | --- | --- || --- | --- |
| 0 | 5.1 | 3.5 | 1.4 |  0.2|  Iris-setosa
| 1 |4.9|  3. |  1.4|  0.2|  Iris-setosa
| 2 |4.7|  3.2|  1.3|  0.2|  Iris-setosa
| 3 |4.6|  3.1|  1.5|  0.2|  Iris-setosa
| 4 |5. |  3.6|  1.4|  0.2|  Iris-setosa


```python
#There are 50 setosa and 50 versicolor and 50 virginica, we are just using setosa and versicolor 
#make the dataset linearly separable
#aka convert the target names to -1 or 1 so we can train with it
data[4] = np.where(data.iloc[:, -1]=='Iris-setosa', 1, -1)

#Convert Pandas dataframe to a Numpy array
np_data = np.asarray(data, dtype = 'float64')

#We will train the Perceptron using the first two attribute sepal length and sepal width
x_train = np_data[:100,:2]
#ouput is the target name which we converted into either 1 or -1
y_train = np_data[:100,-1]

nSamples, dim = x_train.shape
max_iter = 10
acc_ave = 0
acc = 0

w = ####To Do! 
b = ####To Do! 

w_ave = ####To Do! 
b_ave = ####To Do! 

#We'll log the accuracy to plot later 
log_acc = []
log_acc_ave = []

#setting the random seed will generate the same sequence on every run, making it possible to reproduce randomization
np.random.seed(10) 
#Perform a number of iterations over the whole dataset
####To Do! Create a Loop that will perform the inner loop max_iter times!
    c = ####To Do! Initialise a counter to 1
    rnd_idx = ####To Do! #create an array of random indices created at the start of every epoch
    ####To Do! Create a Loop that will iterate over rnd_idx returning one index at a time!
        
        #Sample a single datapoint
        x_i = ####To Do! Use the current index to select the current input TRAINING data point
        y_i = ####To Do! Use the current index to select the current output TRAINING data point
        
        #Calculate the prediction
        y_hat = ####To Do! 
        
        #Only update the Perceptron if the prediction is incorrect
        if ####To Do! Check if the Prediction is correct!
            #if prediction is not correct update the ave-weights with the old weights
            #before updating the weights
            
            w_ave += ####To Do! 
            b_ave += ####To Do! 
            
            #Update the weights and bias
            w += ####To Do! 
            b += ####To Do! 
        
            #reset counter
            c = ####To Do! 
        else:
            #if prediction is correct increment counter
            c = ####To Do! 
                    
    #Calculate the output for every sample using the average weights a bias
    y_hat_X_ave = ####To Do! 
    #Compare to the real labels to calculate the accuracy
    acc_ave = ####To Do! 
    log_acc_ave.append(acc_ave)
    
    #Calculate the output for every sample
    y_hat_X = ####To Do! 
    #Compare to the real labels to calculate the accuracy
    acc = ####To Do! 
    log_acc.append(acc)
    
    print("Iter{0}: accuracy: {1:.3f}, averaged accuracy: {2:.3f}".format(cur_iter+1, acc, acc_ave))  
```

> Jupyter-Notebooks: [Linear_Regression_For_Classification](http://localhost:8888/notebooks/pytorch-tutorial/section1_numpy_ml/notebooks/Linear_Regression_For_Classification.ipynb), [Averaged_Perceptron](http://localhost:8888/notebooks/pytorch-tutorial/section1_numpy_ml/notebooks/Averaged_Perceptron.ipynb), [Numpy_KMeans](http://localhost:8888/notebooks/pytorch-tutorial/section1_numpy_ml/notebooks/Numpy_KMeans.ipynb)

