---
layout: post
title: Section 2, PyTorch - Basics, Neural Networks
author: Alessandro Santospirito
---

<!-- <div style="display: flex; align-items: center; justify-content: space-between;">
  <h2 id="pytorch--basics">PyTorch - Basics</h2>
  <div class='toggle' id='switch'>
    <div class='toggle-text-off'>Markdown</div>
    <div class='glow-comp'></div>
    <div class='toggle-button'></div>
    <div class='toggle-text-on'>Streamlit</div>
  </div>
</div>

<div id="root">
    <iframe id="iframeContent" src="http://localhost:4000/public/html/pytorch-basics.html" style="height: 1000px; width: 100%; display: none; border: none;"></iframe>
</div> -->

<div id="markdownContent" markdown="1">

### nn.Module
```python
class LinearModel(nn.Module):
    """
    Takes the input (x) and returns x * w^t + b
    """
    def __init__(self, input_size, output_size):
        # Pass our class and self to superclass and call the superclass's init function
        super(LinearModel, self).__init__() 
        # nn.Parameter wraps our normal tensors and "tells" Pytorch
        # that they are our nn.Module's model parameters to be optimized 
        self.w = nn.Parameter(torch.randn(output_size, input_size))
        self.b = nn.Parameter(torch.randn(1, output_size))

    def forward(self, x):
        return torch.matmul(x,  self.w.t()) + self.b
```
#### Preperation
```python
# Create a batch of 10 datapoints each 5D
input_data = torch.randn(10, 5)

# Create an instance of our Model
linear_model = LinearModel(5, 1)

# Perform a forward pass!
output = linear_model(input_data)

# Create a random data input tensor
data = torch.randn(100, 3)
# Create some noisey target data
target = data.sum(1, keepdims=True) + 0.01*torch.randn(data.shape[0], 1)
```

<img src="{{ site.url }}/images/2_pytorch-basics/predictions-before-training.png" alt="predictions before training" style="width: 45%;"/>

#### Train Loss
```python
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

loss = loss_function(target_pred, target)
loss.backward()
optimizer.step()

# Perform another forward pass of the model to check the new loss
target_pred = linear(data)
loss = loss_function(target_pred, target)

# Lets create an empty array to log the loss
loss_logger = []

# Lets perform 100 itterations of our dataset
for i in range(1000):
    # Perform a forward pass of our data
    target_pred = linear(data)
    
    # Calculate the loss
    loss = loss_function(target_pred, target)
    
    # .zero_grad sets the stored gradients to 0
    # If we didn't do this they would be added to the 
    # Gradients from the previous step!
    optimizer.zero_grad()
    
    # Calculate the new gradients
    loss.backward()
    
    # Perform an optimization step!
    optimizer.step()

    loss_logger.append(loss.item())
    
```
<div style="display: flex; justify-content: space-between;">
    <img src="{{ site.baseurl }}/images/2_pytorch-basics/loss-logger.png" alt="logging of the loss" style="width: 45%;"/>
    <img src="{{ site.baseurl }}/images/2_pytorch-basics/predictions-after-training.png" alt="predictions after training" style="width: 45%;"/>
</div>

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