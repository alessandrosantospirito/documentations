import streamlit as st

st.title("PyTorch-Basics")

st.markdown(f'''
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

#Randomly generate K indicies for k datapoints from the dataset (indicies need to be int)
means  = test_x_tensor[np.random.randint(0, num_img , num_means)]

plt.figure(1, figsize=(20, 10))
img = means.cpu().view(num_means*28,28)
plt.imshow(img)
```
''')
