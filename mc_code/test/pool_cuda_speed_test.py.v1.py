from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray
import numpy as np

drv.init()

import threading, Queue, time

class gpu_thread(threading.Thread):
	def __init__(self, gpu_number):
		threading.Thread.__init__(self)
		
		self.gpu_number = gpu_number

		print '\n%d gpus found\n' % (drv.Device.count())

		start_time = time.time()
		
		print 'Time to create context = %.3e' % (time.time() - start_time)


	def set_args_list(self, args_list):
		# will be passed list of args list
		# for multiple iterations
		self.__args_list = args_list


	def set_kwargs(self, kwargs):
		self.__kwargs = kwargs


	def set_target(self, target_func):
		self.__target = target_func



	def run(self):
		self.dev = drv.Device(self.gpu_number)
		#self.ctx = self.dev.make_context(drv.ctx_flags.SCHED_AUTO | drv.ctx_flags.MAP_HOST)
		self.ctx = self.dev.make_context()
		start_time_compiler = time.time()
		self.observables_func = pycuda.compiler.SourceModule(cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production')
		print 'Time to compile: %.2e' % (time.time() - start_time_compiler)
		print self.gpu_number
		#print len(self.__args_list)
		for args in self.__args_list:
			self.observables_func(*args, grid=(1024, 1), block=(128, 1, 1))
			#self.__target(self.observables_func, *args, **self.__kwargs)
			#print np.asarray(args[2])

		#self.ctx.pop()





class gpu_pool(object):
	def __init__(self, num_gpus):
		self.num_gpus = num_gpus
		
		self.l_threads = [gpu_thread(i) for i in xrange(self.num_gpus)]

		self.kwargs = {}

	def set_kwargs(self, kwargs={}):
		self.kwargs = kwargs


	def map(self, function_to_call, arguments_for_func_call):
		
		start_time_map = time.time()
		l_thread_tasks = [[] for i in xrange(len(arguments_for_func_call))]
		for i in xrange(len(arguments_for_func_call)):
			l_thread_tasks[i%self.num_gpus].append(arguments_for_func_call[i])

		num_gpus_needed = self.num_gpus
		if self.num_gpus > len(arguments_for_func_call):
			num_gpus_needed = len(arguments_for_func_call)

		# set args lists for each thread
		for i in xrange(num_gpus_needed):
			self.l_threads[i].set_args_list(l_thread_tasks[i])
			self.l_threads[i].set_kwargs(self.kwargs)
			self.l_threads[i].set_target(function_to_call)
		print 'map initialization time: %.2e' % (time.time() - start_time_map)
		
		for i in xrange(num_gpus_needed):
			self.l_threads[i].start()

		for i in xrange(num_gpus_needed):
			self.l_threads[i].join()



cuda_full_observables_production_code ="""
#include <curand_kernel.h>

extern "C" {

__device__ int gpu_binomial(curandState_t *rand_state, int num_successes, float prob_success)
{
	int x = 0;
	for(int i = 0; i < num_successes; i++) {
    if(curand_uniform(rand_state) < prob_success)
		x += 1;
	}
	return x;
}

__global__ void gpu_full_observables_production(int *seed, int *num_trials, float *aS1, float *aS2, float *aEnergy, float *photonYield, float *chargeYield, float *excitonToIonRatio, float *g1Value, float *extractionEfficiency, float *gasGainValue, float *gasGainWidth, float *speRes, float *intrinsicResS1, float *intrinsicResS2)
{

	// start random number generator
	curandState s;
	//const int iteration = blockIdx.x * blockDim.x + threadIdx.x;
	const int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	//curand_init(0, 0, 0, &s); // for debugging
	curand_init(*seed % iteration, 0, 0, &s);
	
	float probRecombination = (( (*excitonToIonRatio+1) * *photonYield )/(*photonYield+*chargeYield) - *excitonToIonRatio);

	float mcEnergy;
	int mcQuanta;
	float probExcitonSuccess;
	int mcExcitons;
	int mcIons;
	int mcRecombined;
	int mcPhotons;
	int mcElectrons;
	int mcExtractedElectrons;
	float mcS1;
	float mcS2;
	
	if (iteration < *num_trials)
	{
	
		// ------------------------------------------------
		//  Draw random energy from distribution
		// ------------------------------------------------
		
		
		mcEnergy = aEnergy[iteration];
		//aS1[iteration] = mcEnergy;
		//return;
		
		if (mcEnergy < 0) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		

		// ------------------------------------------------
		//  Find number of quanta
		// ------------------------------------------------
		
		
		mcQuanta = curand_poisson(&s, mcEnergy*(*photonYield + *chargeYield));
		//aS1[iteration] = mcQuanta;
		//return;
		
		
		// ------------------------------------------------
		//  Convert to excitons and ions
		// ------------------------------------------------
		
		
		probExcitonSuccess = 1. - 1./(1. + *excitonToIonRatio);
		if (probExcitonSuccess < 0 || probExcitonSuccess > 1) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		mcExcitons = gpu_binomial(&s, mcQuanta, probExcitonSuccess);
		mcIons = mcQuanta - mcExcitons;
		//aS1[iteration] = mcExcitons;
		//aS2[iteration] = mcIons;
		//return;
		
		// ------------------------------------------------
		//  Ion recombination
		// ------------------------------------------------

		if (mcIons < 1 || probRecombination < 0 || probRecombination > 1) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		mcRecombined = gpu_binomial(&s, mcIons, probRecombination);
		mcPhotons = mcExcitons + mcRecombined;
		mcElectrons = mcIons - mcRecombined;
		//aS1[iteration] = mcPhotons;
		//aS2[iteration] = mcElectrons;
		//return;
		
		
		// ------------------------------------------------
		//  Convert to S1 and S2 BEFORE smearing
		// ------------------------------------------------
		
		if (mcPhotons < 1 || *g1Value < 0 || *g1Value > 1) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		if (mcElectrons < 1 || *extractionEfficiency < 0 || *extractionEfficiency > 1) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		if (*gasGainWidth <= 0) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		mcS1 = gpu_binomial(&s, mcPhotons, *g1Value);
		mcExtractedElectrons = gpu_binomial(&s, mcElectrons, *extractionEfficiency);
		mcS2 = (curand_normal(&s) * *gasGainWidth*powf(mcExtractedElectrons, 0.5)) + mcExtractedElectrons**gasGainValue;
		
		if (mcS1 < 0) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		if (mcS2 < 0) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		//aS1[iteration] = mcS1;
		//aS2[iteration] = mcS2;
		//return;

		
		
		// ------------------------------------------------
		//  Smear S1 and S2
		// ------------------------------------------------
		
		if (*speRes <= 0 || *intrinsicResS1 <= 0 || *intrinsicResS2 <= 0) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		mcS1 = (curand_normal(&s) * *speRes*powf(mcS1, 0.5)) + mcS1;
		if (mcS1 < 0) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		mcS1 = (curand_normal(&s) * *intrinsicResS1*mcS1) + mcS1;
		if (mcS1 < 0) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		
		mcS2 = (curand_normal(&s) * *intrinsicResS2*mcS2) + mcS2;
		if (mcS2 < 0) 
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		aS1[iteration] = mcS1;
		aS2[iteration] = mcS2;
		
	
	}

  
}

}
"""


g_pool = gpu_pool(2)

#observables_func = pycuda.compiler.SourceModule(cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production')

grid_dim = 2048
block_dim = 256
num_entries = grid_dim*block_dim
num_iterations = 1000


aEnergy = np.full(num_entries, 10., dtype=np.float32)

aS1 = np.full(num_entries, -1, dtype=np.float32)
aS2 = np.full(num_entries, -1, dtype=np.float32)

seed = np.asarray(int(time.time()*1000), dtype=np.int32)
num_trials = np.asarray(num_entries, dtype=np.int32)
photonYield = np.asarray(5, dtype=np.float32)
chargeYield = np.asarray(5, dtype=np.float32)
excitonToIonRatio = np.asarray(.5, dtype=np.float32)
g1Value = np.asarray(.12, dtype=np.float32)
extractionEfficiency = np.asarray(.9, dtype=np.float32)
gasGainValue = np.asarray(30., dtype=np.float32)
gasGainWidth = np.asarray(5., dtype=np.float32)
speRes = np.asarray(.7, dtype=np.float32)
intrinsicResS1 = np.asarray(.3, dtype=np.float32)
intrinsicResS2 = np.asarray(.1, dtype=np.float32)

tArgs = [drv.In(seed), drv.In(num_trials), drv.InOut(aS1), drv.InOut(aS2), drv.In(aEnergy), drv.In(photonYield), drv.In(chargeYield), drv.In(excitonToIonRatio), drv.In(g1Value), drv.In(extractionEfficiency), drv.In(gasGainValue), drv.In(gasGainWidth), drv.In(speRes), drv.In(intrinsicResS1), drv.In(intrinsicResS2)]

def f_gpu_observables_func(func, seed, num_trials, aS1, aS2, aEnergy, photonYield, chargeYield, excitonToIonRatio, g1Value, extractionEfficiency, gasGainValue, gasGainWidth, speRes, intrinsicResS1, intrinsicResS2):

	tArgs = [drv.In(seed), drv.In(num_trials), drv.InOut(aS1), drv.InOut(aS2), drv.In(aEnergy), drv.In(photonYield), drv.In(chargeYield), drv.In(excitonToIonRatio), drv.In(g1Value), drv.In(extractionEfficiency), drv.In(gasGainValue), drv.In(gasGainWidth), drv.In(speRes), drv.In(intrinsicResS1), drv.In(intrinsicResS2)]


	func(*tArgs, grid=(2048, 1), block=(256, 1, 1))

#g_pool.set_kwargs({')
start_time = time.time()
g_pool.map(f_gpu_observables_func, [tArgs for i in xrange(num_iterations)])
print '\nTime for %d iterations: %.2e s' % (num_iterations, time.time() - start_time)


#g_pool.map(f_gpu_practice, [[a_gpu.gpudata], [b_gpu.gpudata], [c_gpu.gpudata], [d_gpu.gpudata]])


#g_pool.map(f_hello, [[1], [2], [3], [4], [5], [6]])
#g_pool.map(f_hello, [[11], [12], [13], [14], [15], [16]])
