from rootpy.plotting import Hist2D, Hist, Legend, Canvas
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

import numpy as np


l_function_calls = [1024*64, 1024*128, 1024*256, 2048*256, 2048*512, 4096*512, 8192*512]

# time is to complete 10 iterations of the number of function calls
# function calls is the number of MC events produced per likelihood
# calculation
l_c_time = [0.867104, 1.701473, 3.387599, 7.079457, 15.393931, 27.637270, 55.280885]
l_stock_gpu_time = [0.106817, 0.204687, 0.289345, 0.393346, 0.813838, 1.281815, 2.214502]
l_lukes_gpu_time = [0.013091, 0.024379, 0.047134, 0.090737, 0.201780, 0.408195, 0.811484]
l_upgraded_gpu = [0.016722, 0.020058, 0.027428, 0.045392, 0.082667, 0.163333, 0.299870] # 121 s on upgraded CPU
#l_gtx_1080 = [/10., /10., /10., /10., /10., /10., 1.953962/10.]


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

# upgraded GPU: GeForce GTX 970
# core config = 1664*104*56
# cache = 4 GB
# GFLOPS = 3494
# bandwith = 200 GB/s
# release: Nov 2014

l_speed_increase_stock = np.divide(l_c_time, l_stock_gpu_time)
l_speed_increase_luke = np.divide(l_c_time, l_lukes_gpu_time)
l_speed_increase_upgraded = np.divide(l_c_time, l_upgraded_gpu)

l_million_function_calls = np.asarray(l_function_calls)/1.e6

fig = plt.figure()
ax = fig.add_subplot(111)

p_stock = ax.plot(l_million_function_calls, l_speed_increase_stock, 'ro', label='GT 650M (Stock - Mar 2012)')
p_luke = ax.plot(l_million_function_calls, l_speed_increase_luke, 'bo', label='GTX 480 (Upgraded Consumer Level - Mar 2010)')
p_upgraded = ax.plot(l_million_function_calls, l_speed_increase_upgraded, 'go', label='GTX 970 (Upgraded Consumer Level - Nov 2014)')

ax.set_title('CPU vs GPU Speed Comparison')
ax.set_xlabel('Millions of Events through MC')
ax.set_ylabel('Speed increase relative to CPU')

ax.legend(loc='center right')



l_function_calls_mismatched_gpus = [4096*512, 8192*512, 16384*512, 32768*512]
l_million_function_calls_mismatched_gpus = np.asarray(l_function_calls_mismatched_gpus)/1.e6

l_upgraded_gpu_only = [3.79, 6.08, 10.571463, 19.47]
l_upgraded_with_very_old_gpu = [3.43, 5.516, 9.551, 17.73]

l_speed_increase_mismatched = np.divide(l_upgraded_gpu_only, l_upgraded_with_very_old_gpu)

l_function_calls_matched_gpus = [2048*1024, 4096*1024, 8192*1024, 16384*1024, 32768*1024, 65536*1024]
l_million_function_calls_matched_gpus = np.asarray(l_function_calls_matched_gpus)/1.e6
l_single_970 = [9.916, 19.13, 37.66, 74.81, 148.4, 295.8]
l_gtx_1080 = [3.155, 5.126, 10.56, 21.89, 37.53, 73.29]
l_dual_970 = [5.209, 10.72, 20.44, 40.27, 81.43, 160.9]
# both done with 500 iterations each

l_speed_increase_matched = np.divide(l_single_970, l_dual_970)

fig_2 = plt.figure()
ax_2 = fig_2.add_subplot(111)

p_combined = ax_2.plot(l_million_function_calls_mismatched_gpus, l_speed_increase_mismatched, 'bo', label='GTX 970 with GT 430 vs. GTX 970 only')
p_dual_970 = ax_2.plot(l_million_function_calls_matched_gpus, l_speed_increase_matched, 'ro', label='Dual GTX 970s vs. GTX 970 only')
ax_2.set_ylim([0.9, 2.0])

ax_2.set_title('GPU Parallel Study')
ax_2.set_xlabel('Millions of Events through MC')
ax_2.set_ylabel('(Time with GTX 970 only) / (Time with GTX 970 and GT 430 in parallel)')

fig_3, ax_3 = plt.subplots(1)
print l_million_function_calls_matched_gpus, np.divide(l_single_970, l_gtx_1080)
ax_3.plot(l_million_function_calls_matched_gpus, np.divide(l_single_970, l_gtx_1080), 'bo')

ax_3.set_title('Relative Speed of GTX 1080 and 970')
ax_3.set_xlabel('Millions of Events through MC')
ax_3.set_ylabel('(Time for GTX 970) / (Time for GTX 1080)')

plt.show()


