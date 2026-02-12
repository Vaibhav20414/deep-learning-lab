#why do we use the data type we use in while using pytorch,
#what are the possible reason-
# we want to extract the following information while using a datatype-
# 1. How many bits will be used per number 2. How will we interpret those bits,either it's going to be int or float
# 3. How fast can we do maths on device, device being CPU/GPU 4. How accurate the gradient will be
# this will be a tradeoff between, Precision x Numerical stabiliy x Speed x Memory, all the component can be understood with the 4 points above

import torch

x = torch.tensor([1.0]) #what is in the square bracket, is it a list, tuple or something else.
print(x.dtype) #here the default datatype is float32
#but why does this is the datatype of x, not of the thing in inside the bracket

#float32 is a safe middle ground, which provide enough precision for the computation of gradient and backpropogation 
#while preserving the memory efficiency 

#if we were to try 

y = torch.tensor([1.0], dtype=torch.int32, requires_grad=True)
print(y)

#torch will throw an error - only tensor of floating point and complex dtype can require gradients
#reason -
# Learning and gradients require small changes, which are captured by decimal, these decimal values are used for updation 

#in the line 10 we used a basic python data structure like list, this basic ds is copied to the tenson after 
#allocation of contigious memory in CPU/GPU and inferring of data type (defaut is mentioned above) 
#we can use other datatype like a numpy array, tuple or nested list(for 2d matrices)
# SHARING MEMORY SPACE USING - torch.from_numpy()

