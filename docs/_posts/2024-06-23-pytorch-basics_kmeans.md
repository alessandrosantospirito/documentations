---
layout: post
title: Section 2, PyTorch - Basics, K-Means Condition
author: Alessandro Santospirito
---

<div id="conditionalContent" style="display: flex; align-items: center; justify-content: space-between;">
  <h2 id="pytorch--basics">K-Means Condition</h2>
  <div class='toggle' id='switch'>
    <div class='toggle-text-off'>Markdown</div>
    <div class='glow-comp'></div>
    <div class='toggle-button'></div>
    <div class='toggle-text-on'>Streamlit</div>
  </div>
</div>

<div id="root">
    <iframe id="iframeContent" src="http://localhost:4000/public/html/pytorch-basics_kmeans.html" style="height: 1000px; width: 100%; display: none; border: none;"></iframe>
</div>

<div id="markdownContent" markdown="1">

![](../../../../images/2_pytorch-basics/K_means.gif)

#### Description
Lets have a look at the steps of K-means clustering
1. Define the number of clusters "k" you want to group your data into
2. Randomly initialise k vectors with the same size as each datapoint, this is the initialisation of our cluster centers
3. Calculate the distance between each datapoint and each cluster center (using MSE or equivalent)
4. For every datapoint find the cluster center they are closest to
5. Re-calculate the cluster centers by finding the mean of every new cluster
6. Repeat steps 3-6 for n steps or until convergence

#### Code
```python
#Number of datapoint
num_img = 10000  
#Number of cluster centers, 10 because the dataset contains 10 classes eg: digit 0 to 9
num_means = 10   
#We'll perform this many iterations of the algorithm
iterations = 20 
#Each image is 28*28 pixels, which has been flattened to a vector 0f 784 values
data_size = 28*28
# The images are 8 bit greyscale images (values range from 0-255)
# We'll rescale the pixel values to be between 0-1 (We don't REALLY need to do this for k-means)
test_x_tensor = torch.FloatTensor((test_x.astype(float) / 255))

#Radnomly generate K indicies for k datapoints from the dataset (indicies need to be int)
means  = test_x_tensor[np.random.randint(0, num_img , num_means)]
```

## Gradient Decent
![](../../../../images/1_pytorch-basics/Linear_Regression.gif)

#### Loading Data
```python
# You can load your data using this cell
npzfile = np.load("../data/toy_data_two_moon.npz") # toy_data.npz or toy_data_two_circles.npz

# The compressed Numpy file is split up into 4 parts
# Lets convert them to Pytorch Float Tensors
# Train inputs and target outputs
x_train = torch.FloatTensor(npzfile['arr_0'])
y_train = torch.FloatTensor(npzfile['arr_2'])

# Test inputs and target outputs
x_test = torch.FloatTensor(npzfile['arr_1'])
y_test = torch.FloatTensor(npzfile['arr_3'])
```

#### Training a model with GD
<h2>Training a model with GD </h2>

[Gradient Descent, Step-by-Step by StatQuest](https://youtu.be/sDv4f4s2SB8?si=iClqYh2v3I7uf9WR)

In doing so, we need a function to <br>
1- compute the loss with respect to the inputs and the parameters of the model <br>
2- compute the gradient of the model with respect to its parameters $\theta$

We recall the loss of the linear regression as

$$
\begin{align}
L(\theta) = \frac{1}{m} \sum_{i=1}^m \|\theta^\top \boldsymbol{x}_i - y_i\|^2
\end{align}
$$

Now it is easy to see that

$$
\begin{align}
\frac{\partial L}{\partial \theta} = \frac{1}{m} \sum_{i=1}^m 2(\theta^\top \boldsymbol{x}_i - y_i)\boldsymbol{x}_i
\end{align}
$$

Instead of calculating the gradient by hand, we'll just use Pytorch's auto-grad!!

```python
# Define our linear model - 2 inputs, 1 output (bias is included in linear layer)
linear = # To Do
loss_function = # To Do
optimizer = # To Do
max_epoch = 100

loss_log = [] # keep track of the loss values
acc = [] # keep track of the accuracy 
for epoch in range(max_epoch):
    
    # Perform a test set accuracy calculation
    with torch.no_grad():
        y_test_hat = # To Do
        class_pred = # To Do
        acc.append(# To Do)

    # Perform a training step
    
    # Forward pass
    y_train_hat = # To Do
            
    # Calculate loss
    loss = # To Do
    
    # Zero the gradient
    # To Do
            
    # Perform backprop
    # To Do
    
    # Perform optimization step
    # To Do
    
    # Append loss
    loss_log.append(# To Do)
```

> Jupyter-Notebooks: [Pytorch-Basics](http://localhost:8888/notebooks/pytorch-tutorial/section2_pytorch_basics/notebooks/Tutorial1_Pytorch_Basics.ipynb), [Kmeans Clustering](http://localhost:8888/notebooks/pytorch-tutorial/section2_pytorch_basics/notebooks/Pytorch1_KMeans.ipynb), [Gradient Descent Revisit with PyTorch](http://localhost:8888/notebooks/pytorch-tutorial/section2_pytorch_basics/notebooks/Pytorch2_Linear_Logistic_Regression_For_Classification.ipynb)
</div>

<script>
  document.addEventListener("DOMContentLoaded", function() {
    if (window.location.href.includes("4000/2024")) {
    } else{
      document.getElementById("conditionalContent").style.display = "none";
    }
  });
  document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.querySelector('.toggle');
    toggle.addEventListener('click', function(e) {
      e.preventDefault();
      this.classList.toggle('toggle-on');
      updateToggleState(this);
    });
  
    function updateToggleState(toggleElement) {
      const isOn = toggleElement.classList.contains('toggle-on');
  
      var iframeContent = document.getElementById('iframeContent');
      var markdownContent = document.getElementById('markdownContent');
      if (isOn) {
        iframeContent.style.display = '';
        markdownContent.style.display = 'none';
      } else {
        iframeContent.style.display = 'none';
        markdownContent.style.display = '';
      }
    }
  });
</script>