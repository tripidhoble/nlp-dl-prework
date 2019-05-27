# --------------
# Code starts here

import numpy as np

# Code starts here

# Adjacency matrix
adj_mat = np.array([[0,0,0,0,0,0,1/3,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                   [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                  [0,0,1/2,1/3,0,0,1/3,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/3,0]])

# Compute eigenvalues and eigencevectrs
eigenvalues, eigencevectors = np.linalg.eig(adj_mat)
print("eigenvalues: ", eigenvalues)
print("eigencevectors: ", eigencevectors)

# Eigen vector corresponding to 1
eigen_1 = abs(eigencevectors[:,0]) / np.linalg.norm(eigencevectors[:,0], 1)

# most important page
page = np.where(np.max(eigen_1) == eigen_1)[0][0] + 1

# Code ends here


# --------------
# Code starts here

# Initialize stationary vector I
init_I = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
for i in range(10):
     init_I = np.dot(adj_mat, init_I) / np.linalg.norm(init_I, 1)
     
power_page = np.where(np.max(init_I) == init_I)[0][0] + 1

# Code ends here


# --------------
# Code starts here

# New Adjancency matrix
# New Adjancency matrix
new_adj_mat = np.array([[0,0,0,0,0,0,0,0],
                   [1/2,0,1/2,1/3,0,0,0,0],
                  [1/2,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0],
                   [0,0,1/2,1/3,0,0,1/2,0],
                   [0,0,0,1/3,1/3,0,0,1/2],
                   [0,0,0,0,1/3,0,0,1/2],
                   [0,0,0,0,1/3,1,1/2,0]])

# Initialize stationary vector I
new_init_I  = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
for i in range(10):
    new_init_I = np.dot(new_adj_mat, new_init_I) / np.linalg.norm(new_init_I, 1)

new_page_rank = np.where(np.max(new_adj_mat) == new_adj_mat) [0][0] + 1
print("new_init_I: ", new_init_I)
print("new_page_rank: ", new_page_rank)



# Code ends here


# --------------
# Alpha value
alpha = 0.85

# Code starts here

# Modified adjancency matrix
G = alpha * new_adj_mat + (np.ones(new_adj_mat.shape) - alpha) * (1/ len(new_adj_mat))

# Initialize stationary vector I
final_init_I = np.array([1,0,0,0,0,0,0,0])

# Perform iterations for power method
for i in range(10):
    final_init_I = np.dot(new_adj_mat, final_init_I) / np.linalg.norm(final_init_I, 1)

final_page_rank = np.where(np.max(final_init_I) == final_init_I)[0][0] + 1

print("final_init_I: ", final_init_I)
print("final_page_rank: ", final_page_rank)

# Code ends here


