import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

#Python script for plotting the performance.

def save(name):
    fig.savefig('{}.eps'.format(name), format='eps', dpi=1200, bbox_inches='tight')


#input_filename = sys.argv[1]
#perf_filenames = sys.argv[2:]

input_sizes = np.array([200, 2000, 20000, 200000, 2e6, 2e7])

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel('Input size')
ax.set_ylabel('[GFlops/s]', rotation=0, ha='left')
ax.set_title('Performance', loc='left', fontdict={'weight':'bold'}, pad=25)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_label_coords(0, 1.02)

ax.set_facecolor((0.9, 0.9, 0.9))
ax.set_yticks(np.arange(0, 11, 1))
plt.grid(color='w', axis='y')
plt.ylim([0, 10])

# log-scale x
ax.set_xscale('log', basex=10)
plt.xticks(input_sizes)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
ax.margins(x=0.04)
# my values, insert your own.
FPerf = [3, 3.45, 2.75, 2.55, 2.3, 2.28]
Ftrans = [2, 2, 1.55]
Fomp = [0.7, 3.5, 6.14, 3.5, 2.6]
fintel = [0.2, 1.67, 5.06, 4.12, 5, 5.3]
fnivida = [0.6, 5, 7.77, 8.76, 9.2, 9.1]
matlab = [6.7*10**-4, 6.97*10**-4]
matlabimp = [9*10**-3, 6.4*10**-2]

#lineM, = ax.plot(input_sizes[:2], matlab, '-o', label='MATLAB')
lineF, = ax.plot(input_sizes, FPerf, '-o', label='Fortran')
lineMi, = ax.plot(input_sizes[:2], matlabimp, '-o', label='MATLAB')
lineT, = ax.plot(input_sizes[:3], Ftrans, '-o', label='Fortran trans')
lineO, = ax.plot(input_sizes[:-1], Fomp, '-o', label='Fortran OpenMP')
lineI, = ax.plot(input_sizes, fintel, '-o', label='OpenCL Intel')
lineN, = ax.plot(input_sizes, fnivida, '-o', label='OpenCL NVIDIA')
#line.set_clip_on(False)


plt.legend()
plt.show()
