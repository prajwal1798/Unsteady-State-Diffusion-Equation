#!/usr/bin/env python
# coding: utf-8

# ## STEP 3 :       1-D Unsteady State Diffusion Equation

# Lets start by importing necessary libraries 

# In[158]:


import numpy as np
import matplotlib.pyplot as plt


# Defining the domain for both Space & Time 

# In[159]:


xmax  = 2                         # Total Domain lenth in (m)
nx    = 61                        # Number of Grid Points in space 'X'
dx    = xmax/(nx-1)               # Size of each grid in (m)
nt    = 20                        # Number of Grid Points in time 't'
alpha = 0.4                       # CFL Number
nu    = 0.3                       # Viscosity in m^2/s
dt    = alpha*dx**2/nu            # Time-step size in (s)
x     = np.linspace(0,xmax,nx)    # Space Domain with grids in (m)


# Determining the CFL Number  $\alpha$:
# 
# $ \alpha = \nu * \frac{\Delta t}{\Delta x^2} $
# 
# CFL Criteria for the below criteria of Schemes used; 
# 
# $ \alpha \leq 0.5 $
# 
# i.e, $ \nu * \frac{\Delta t}{\Delta x^2} \leq 0.5 $ 
# 
# Hence we have chosen $\alpha \leq 0.5$
#  

# Defining the Initial Conditions

# In[160]:


u = np.ones(nx)                      # Initialise the velocity array of ones
u[int(.5/dx):int((1/dx)+1)] = 2      # Implementing the square wave condition for velocity


# Plotting the Velocity field function U to understand its profile variation in Space

# In[161]:


plt.plot(x,u)
plt.title('Velocity Profile - Square Wave')
plt.xlabel('X (m)')
plt.ylabel('u (m/s)')
plt.grid()
plt.show


# Discretisation of Diffusion equation & finding its max grid resolution to avoid solution blow-up  
# 
# $$\frac{\partial u}{\partial t} = \nu * \frac{\partial^2 u}{\partial x^2}$$ 
# 
# Forward Differencing in Time :
# $$\frac{\partial u}{\partial t} \approx \frac{u_i^{n+1}-u_i^{n}}{\Delta t}\rightarrow 1$$
# 
# Backward Differencing in Space :
# $$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1}^{n}+u_{i-1}^{n}-(2*u_i^{n})}{\Delta x^2}\rightarrow 2$$
# 
# Therefore expressing Velocity explicitly, we have;
# $$u_i^{n+1} = u_i^{n} -(\nu *\frac{\Delta t}{\Delta x^2})*({u_{i+1}^{n}+u_{i-1}^{n}-(2*u_i^{n})}) \rightarrow 3$$  

# Lets start by Intialising a new array of velocity $u_n$

# In[162]:


#Initialize a temporary array
un = np.ones(nx) 

# Time Loop
for n in range(nt+1):
    un = u.copy()
# Space Loop
    for i in range(1,nx-1):
        u[i] = un[i] + nu * dt/dx**2 *(un[i+1] - 2*un[i] + un[i-1])


# Plotting the Velocity field function U to understand its profile variation in Space

# In[163]:


plt.plot(x,u)
plt.title('Diffused Profile Velocity')
plt.xlabel('X (m)')
plt.ylabel('u (m/s)')
plt.grid()
plt.show


# In[ ]:




