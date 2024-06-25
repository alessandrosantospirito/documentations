---
layout: post
title: Section 2, PyTorch - Basics, Autograd
author: Alessandro Santospirito
---

<div id="markdownContent" markdown="1">
### Autograd

```python
# Lets create some tensors, requires_grad tells Pytorch we want to store the gradients for this tensor
# we need to do this if we are working with basic Pytorch tensors
x = torch.FloatTensor([4])
x.requires_grad = True
w = torch.FloatTensor([2])
w.requires_grad = True
b = torch.FloatTensor([3])
b.requires_grad = True

# By performing a simple computation Pytorch will build a computational graph.
y = w * x + b    # y = 2 * x + 3

# It's easy to see that
# dy/dx = w = 2
# dy/dw = x = 4
# dy/db = 1

# Compute gradients via Pytorch's Autograd
y.backward()
```

#### Minimum
```python
# Define the equation as a lambda function
fx = lambda  x: x**2 + 1.5 * x - 1

x = np.linspace(-10, 8.5, 100)


# Create a random point X
x_ = torch.randn(1)
x_.requires_grad = True

# Create some loggers
x_logger = []
y_logger = []

counter = 0
learning_rate = 0.01
dy_dx_ = 1000
max_num_steps = 1000

# Keep taking steps untill the gradient is small
while np.abs(dy_dx_) > 0.001:
    # Get the Y point at the current x value
    y_ = fx(x_)
    
    # Calculate the gradient at this point
    y_.backward()
    dy_dx_ = x_.grad.item()

    # Pytorch will not keep track of operations within a torch.no_grad() block
    # We don't want Pytorch to add our gradient decent step to the computational graph!
    with torch.no_grad():
        # Take a step down (decend) the curve
        x_ -= learning_rate * dy_dx_
        
        # Pytorch will accumulate the gradient over multiple backward passes
        # For our use case we don't want this to happen so we need to set it to zero
        # After we have used it
        x_.grad.zero_()
        
        # Log the X and Y points to plot
        x_logger.append(x_.item())
        y_logger.append(y_.item())
        
    counter += 1
    
    if counter == max_num_steps:
        break
```


> Jupyter-Notebooks: [Pytorch-Basics](http://localhost:8888/notebooks/pytorch-tutorial/section2_pytorch_basics/notebooks/Tutorial1_Pytorch_Basics.ipynb), [Kmeans Clustering](http://localhost:8888/notebooks/pytorch-tutorial/section2_pytorch_basics/notebooks/Pytorch1_KMeans.ipynb), [Gradient Descent Revisit with PyTorch](http://localhost:8888/notebooks/pytorch-tutorial/section2_pytorch_basics/notebooks/Pytorch2_Linear_Logistic_Regression_For_Classification.ipynb)
</div>

<script>
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
        iframeContent.style.display = 'block';
        markdownContent.style.display = 'none';
      } else {
        iframeContent.style.display = 'none';
        markdownContent.style.display = 'block';
      }
    }
  });
</script>