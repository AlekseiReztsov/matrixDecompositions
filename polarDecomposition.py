from PIL import Image
import numpy as np
from scipy.linalg import polar
image = Image.open('data/mario.jpg').convert('L')
# image.show()

A = np.asarray(image, dtype=np.uint32)
rank_A = np.linalg.matrix_rank(A)
S = np.matmul(A.transpose(), A)
L_vec, Q = np.linalg.eigh(S)
L_vec = L_vec[::-1]
Q = np.fliplr(Q)

# I = np.matmul(Q.transpose(), Q)
# S_REC = np.linalg.multi_dot([V, np.diag(W), V.transpose()])
# max_err = np.amax(S - S_REC)
SQRT_L = np.diag(np.sqrt(L_vec))
SQRT_S = np.linalg.multi_dot([Q, SQRT_L, Q.transpose()])
U = np.linalg.multi_dot([A, Q, np.linalg.matrix_power(SQRT_L, -1), Q.transpose()])
A_REC = np.matmul(U, SQRT_S)
max_err_A = np.amax(A - A_REC)

U_s = np.matmul(U, Q)
SIGMA = SQRT_L
V_t = Q.transpose()

A_REC = np.linalg.multi_dot([U_s, SIGMA, V_t])

U_s_AUTO, SIGMA_AUTO, V_t_AUTO = np.linalg.svd(A, full_matrices=False)

# U_AUTO, SQRT_S_AUTO = polar(A)
# A_REC_AUTO = np.matmul(U_AUTO, SQRT_S_AUTO)
# max_err_A_AUTO = np.amax(A - A_REC_AUTO)
# max_err_U_U_AUTO = np.amax(U - U_AUTO)

# I = np.matmul(U.transpose(), U)

print(type(A))
print(A.shape)
