import numpy as np
import matplotlib.pyplot as plt

a_spamprob = .5
a_t1 = .5
a_t2 = .5

a_spamerr = np.array([1,1])
a_err = [[.95,.97],[.99,.98]]

a_spv = np.array([1-a_spamprob, a_spamprob])
atr = np.array([[1-a_t1, a_t1],[a_t2, 1 - a_t2]])
a_fullerr = np.multiply(atr, a_err)
a_fullspam = np.multiply(a_spv, a_spamerr)

a_eigs, a_vecs = np.linalg.eig(a_fullerr)
a_vecs_inv = np.linalg.inv(a_vecs)

print(a_vecs)
print(a_vecs_inv @ a_fullerr @ a_vecs)
a_meas = np.array([1,1])
# a_meas = a_fullspam
print(a_vecs_inv @ a_meas)
print(a_fullspam @ a_vecs)

# print("probs: ")

print("--------")

b_spamprob = .9
b_t1 = .8
b_t2 = .4

b_spamerr = np.array([1,1])
# b_err = [[.995,.997],[.99,.998]]
b_err = [[.997,.995],[.999,1]]

b_spv = np.array([1-b_spamprob, b_spamprob])
btr = np.array([[1-b_t1, b_t1],[b_t2, 1 - b_t2]])
b_fullerr = np.multiply(btr, b_err)
b_fullspam = np.multiply(b_spv, b_spamerr)

b_eigs, b_vecs = np.linalg.eig(b_fullerr)
b_vecs_inv = np.linalg.inv(b_vecs)

print(b_vecs_inv @ b_fullerr @ b_vecs)
# b_meas = np.array([1-b_])
b_meas = b_fullspam
print(b_vecs_inv @ b_meas)
print(b_fullspam @ b_vecs)


# print(a_spv)
# print(atr)
# print(a_errmat)


def eval_model(spam, meas, mat, n):
    # meas = np.array([1,1])
    return spam @ np.linalg.matrix_power(mat, n) @ meas

ns = list(range(100))
ssa = [eval_model(a_fullspam, a_meas, a_fullerr, n) for n in ns]
ssb = [eval_model(b_fullspam, b_meas, b_fullerr, n) for n in ns]

print("------")
print([float(x) for x in ssa[:10]])
print([float(x) for x in ssb[:10]])
# ssb = [eval_model(b_fullspa)]
# print(eval_model(a_spv, a_errmat, 1))

plt.plot(ns,ssa)
plt.plot(ns,ssb)
# plt.yscale('log')
plt.show()
