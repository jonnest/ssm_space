################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################

from shapesimilarity import shape_similarity
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from scipy.spatial import procrustes

num = 200

x = np.linspace(1, -1, num=num)

y1 = 2*x**2 + 2
y2 = 0*x  # 2*x**2
z1 = 2*y2**3 + 5*y1**5
z2 = 0*x  # 2*y2**3

shape1 = np.column_stack((x, y1, z1))
shape2 = np.column_stack((x, y2, z2))


# shape1 = np.hstack((shape1))
# print(shape1)

similarity = shape_similarity(shape1, shape2)

# procrustes analysis
mtx1, mtx2, disparity = procrustes(shape1, shape2)
# plt.plot(shape1[:,0], shape1[:,1], shape1[:,2], linewidth=2.0)
# plt.plot(shape2[:,0], shape2[:,1], shape2[:,2], linewidth=2.0)
ax = plt.axes(projection='3d')
ax.scatter3D(shape1[:,0], shape1[:,1], shape1[:,2]) 
ax.scatter3D(shape2[:,0], shape2[:,1], shape2[:,2]) 


plt.title(f'Shape similarity is: {similarity}', fontsize=14, fontweight='bold')
plt.show()


print("Similarity   : ", similarity)
print("\nDisparity  : ", round(disparity,3))

