import numpy as np

# Source Image Dimensions Go Here
dim_x_start = 294.0
dim_y_start = 294.0

# Set Convolution Kernel 1 Info
kernel_1_size = 9.0
kernel_1_stepover = 3.0

# Set Pooling Layer 1 Info
pooling_1_size = 2.0

# Set Convolution Kernel 2 Info
kernel_2_size = 9.0
kernel_2_stepover = 3.0

# Set Pooling Layer 2 Info
pooling_2_size = 2.0

### Math Part ###
# Calculate Dimensions Post-Convolution 1
dim_x_conv1 = (dim_x_start - kernel_1_size)/kernel_1_stepover + 1.0
dim_y_conv1 = (dim_y_start - kernel_1_size)/kernel_1_stepover + 1.0
print("Post Conv 1")
print(dim_x_conv1)
print(dim_y_conv1)

# Calculate Dimensions Post-Pooling 1
dim_x_pool1 = dim_x_conv1/pooling_1_size
dim_y_pool1 = dim_y_conv1/pooling_1_size
print("Post Pool 1")
print(dim_x_pool1)
print(dim_y_pool1)

# Calculate Dimensions Post-Convolution 2
dim_x_conv2 = (dim_x_pool1 - kernel_2_size)/kernel_2_stepover + 1.0
dim_y_conv2 = (dim_y_pool1 - kernel_2_size)/kernel_2_stepover + 1.0
print("Post Conv 2")
print(dim_x_conv2)
print(dim_y_conv2)

# Calculate Dimensions Post-Pooling 2
dim_x_pool2 = dim_x_conv2/pooling_2_size
dim_y_pool2 = dim_y_conv2/pooling_2_size
print("Post Pool 2")
print(dim_x_pool2)
print(dim_y_pool2)

## Round and Climb back up
dim_x_intended = np.round(dim_x_pool2,0)
dim_y_intended = np.round(dim_y_pool2,0)

dim_x_needed = (((dim_x_intended*pooling_2_size - 1)*kernel_2_stepover + kernel_2_size)*pooling_1_size - 1)*kernel_1_stepover + kernel_1_size
dim_y_needed = (((dim_y_intended*pooling_2_size - 1)*kernel_2_stepover + kernel_2_size)*pooling_1_size - 1)*kernel_1_stepover + kernel_1_size

print("Dimensions Needed")
print(dim_x_needed)
print(dim_y_needed)