from rootpy.plotting import Hist2D, Hist, Legend, Canvas
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

import numpy as np


l_function_calls = [1024*64, 1024*128, 1024*256, 2048*256, 2048*512, 4096*512, 8192*512]

# time is to complete 10 iterations of the number of function calls
# function calls is the number of MC events produced per likelihood
# calculation
l_stock_gpu_time = [0.106817, 0.204687, 0.289345, 0.393346, 0.813838, 1.281815, 2.214502]
l_c_time = [0.867104, 1.701473, 3.387599, 7.079457, 15.393931, 27.637270, 55.280885]
l_lukes_gpu_time = [0.013091, 0.024379, 0.047134, 0.090737, 0.201780, 0.408195, 0.811484]

# Matt's GPU: GeForce GT 650M 1024 MB
# core config = 384*32*16
# GPLOPs = 641.3
# bandwith = 28.8 GB/s
# cache = 1024 MB
# release: Mar 2012

# Luke's GPU: GEForce GTX 480
# core config = 480*60*48
# GFLOPs = 1344.96
# cache = 1536 MB
# bandwith = 177.4 GB/s
# release: Mar 2010

# proposed GPU: GeForce GTX 970
# core config = 1664*104*56
# cache = 4 GB
# GFLOPS = 3494
# bandwith = 200 GB/s
# release: Nov 2014

l_speed_increase_stock = np.divide(l_c_time, l_stock_gpu_time)
l_speed_increase_luke = np.divide(l_c_time, l_lukes_gpu_time)

l_million_function_calls = np.asarray(l_function_calls)/1.e6

fig = plt.figure()
ax = fig.add_subplot(111)

p_stock = ax.plot(l_million_function_calls, l_speed_increase_stock, 'ro', label='GT 650M (Stock - Mar 2012)')
p_luke = ax.plot(l_million_function_calls, l_speed_increase_luke, 'bo', label='GTX 480 (Upgraded Consumer Level - Mar 2010)')

ax.set_title('CPU vs GPU Speed Comparison')
ax.set_xlabel('Millions of Events through MC')
ax.set_ylabel('Speed increase relative to CPU')

ax.legend(loc='center right')

plt.show()


