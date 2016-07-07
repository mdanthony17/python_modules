import cuda_full_observables_production
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray

import numpy as np

drv.init()
dev = drv.Device(0)
ctx = dev.make_context(drv.ctx_flags.SCHED_AUTO | drv.ctx_flags.MAP_HOST)

gpu_find_lower_bound = SourceModule(cuda_full_observables_production.cuda_full_observables_production_code, no_extern_c=True).get_function('test_gpu_find_lower_bound')


a_test = np.asarray([3., 4., 5., 6., 7.], dtype=np.float32)
num_trials = np.asarray(len(a_test), dtype=np.int32)
a_test_gpu = pycuda.gpuarray.to_gpu(a_test)
search_value = np.asarray(3.01, dtype=np.float32)
search_index = np.asarray(-1, dtype=np.int32)

gpu_find_lower_bound(drv.In(num_trials), a_test_gpu, drv.In(search_value), drv.Out(search_index), grid=(1, 1), block=(1, 1, 1))

print search_index

ctx.pop()