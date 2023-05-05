"""
    Draw convergence plot
"""

import numpy as np
import matplotlib.pyplot as plt

brca = np.load('trace_deq.npy')
# deq_then_sum_mrna = np.load('m1_no_fuse.npy')
# deq_then_sum_dna = np.load('m2_no_fuse.npy')
# deq_then_sum_mirna = np.load('m3_no_fuse.npy')
# sum_then_deq = np.load('trace_fuse_only.npy')
wt = np.load('weight_tie.npy')

step = 100
plt.figure(dpi=500, figsize=(10,5))
plt.yscale('log')

plt.xlabel('Steps')
plt.ylabel('Difference norm')

plt.plot(np.arange(step),brca[:,:step].mean(0))
# plt.plot(np.arange(step),deq_then_sum_mrna[:,:step].mean(0))
# plt.plot(np.arange(step),deq_then_sum_dna[:,:step].mean(0))
# plt.plot(np.arange(step),deq_then_sum_mirna[:,:step].mean(0))
# plt.plot(np.arange(step),sum_then_deq[:,:step].mean(0))
plt.plot(np.arange(step),wt[:,:step].mean(0))

def cal_bounds(l, step):
    l_mean = l[:,:step].mean(0)
#     l_min = l[:,:step].min(0)
#     l_max = l[:,:step].max(0)
    l_std = l[:,:step].std(0) / np.sqrt(l.shape[0])
    return l_mean - 1.96 * l_std, l_mean + 1.96 * l_std
    

plt.fill_between(np.arange(step), cal_bounds(brca, step)[0], cal_bounds(brca, step)[1],
                 facecolor="blue", # The fill color
                 #color='blue',       # The outline color
                 alpha=0.2)
# plt.fill_between(np.arange(step), deq_then_sum_mrna[:,:step].min(0), deq_then_sum_mrna[:,:step].max(0),
#                  facecolor="orange", # The fill color
#                  #color='orange',       # The outline color
#                  alpha=0.2)
# plt.fill_between(np.arange(step), deq_then_sum_dna[:,:step].min(0), deq_then_sum_dna[:,:step].max(0),
#                  facecolor="green", # The fill color
#                  #color='green',       # The outline color
#                  alpha=0.2)
# plt.fill_between(np.arange(step), deq_then_sum_mirna[:,:step].min(0), deq_then_sum_mirna[:,:step].max(0),
#                  facecolor="red", # The fill color
#                  #color='red',       # The outline color
#                  alpha=0.2)
# plt.fill_between(np.arange(step), sum_then_deq[:,:step].min(0), sum_then_deq[:,:step].max(0),
#                  facecolor="purple", # The fill color
#                  #color='purple',       # The outline color
#                  alpha=0.2)
plt.fill_between(np.arange(step), cal_bounds(wt, step)[0], cal_bounds(wt, step)[1],
                 facecolor="orange", # The fill color
                 #color='blue',       # The outline color
                 alpha=0.2)

plt.grid('both')
plt.legend(['DEQ Fusion', 'Weight-tied Fusion'])
# plt.legend(['DEQ Fusion', f'$f_{{\\theta}}$ - mRNA expression data', f'$f_{{\\theta}}$ - DNA methylation data', f'$f_{{\\theta}}$ - miRNA expression data', f'$f_{{fuse}}$'])
plt.savefig('convergence.pdf')