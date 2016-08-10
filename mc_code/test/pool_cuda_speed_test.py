from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray
import numpy as np

#drv.init()

import threading, time
from multiprocessing import Pool, Queue

testing_variable = 'YOU_THE_BEST'

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

def initialize_thread(q):
	from pycuda.compiler import SourceModule
	import pycuda.driver as drv

	gpu_number = q.get()
	#print gpu_number
	
	#print globals()
	
	#time.sleep((gpu_number+0.1)*0.5)
	drv.init()
	dev = drv.Device(gpu_number)
	ctx = dev.make_context()
	start_time_compiler = time.time()
	
	globals()['observables_func'] = pycuda.compiler.SourceModule(cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production')
	
	print 'Successfully loaded GPU!'
	globals()['gpu_arr_energy'] = pycuda.gpuarray.to_gpu(aEnergy)
	



gpu_numbers = Queue()
for num in [0]:
	gpu_numbers.put(num)



def f_gpu_observables_func(tArgs):
	#print 'inside f_gpu...'
	
	aS1 = np.full(num_entries, -1, dtype=np.float32)
	aS1_gpu = pycuda.gpuarray.to_gpu(aS1)
	aS2 = np.full(num_entries, -1, dtype=np.float32)
	aS2_gpu = pycuda.gpuarray.to_gpu(aS2)

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
	
	
	tArgs = [drv.In(seed), drv.In(num_trials), aS1_gpu.gpudata, aS2_gpu.gpudata, gpu_arr_energy.gpudata, drv.In(photonYield), drv.In(chargeYield), drv.In(excitonToIonRatio), drv.In(g1Value), drv.In(extractionEfficiency), drv.In(gasGainValue), drv.In(gasGainWidth), drv.In(speRes), drv.In(intrinsicResS1), drv.In(intrinsicResS2)]

	observables_func(*tArgs, grid=(4096, 1), block=(512, 1, 1))

	print aS1


grid_dim = 4096
block_dim = 512
num_entries = grid_dim*block_dim
aEnergy = np.full(num_entries, 10., dtype=np.float32)

g_pool = Pool(processes=1, initializer=initialize_thread, initargs=(gpu_numbers,))


#observables_func = pycuda.compiler.SourceModule(cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production')


num_iterations = 1000


aS1 = np.full(num_entries, -1, dtype=np.float32)
#aS1_gpu = pycuda.gpuarray.to_gpu(aS1)
aS2 = np.full(num_entries, -1, dtype=np.float32)
#aS2_gpu = pycuda.gpuarray.to_gpu(aS2)

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

tArgs = [drv.In(seed), drv.In(num_trials), drv.InOut(aS1), drv.InOut(aS2), drv.In(photonYield), drv.In(chargeYield), drv.In(excitonToIonRatio), drv.In(g1Value), drv.In(extractionEfficiency), drv.In(gasGainValue), drv.In(gasGainWidth), drv.In(speRes), drv.In(intrinsicResS1), drv.In(intrinsicResS2)]

"""
def f_gpu_observables_func(seed, num_trials, aS1, aS2, aEnergy, photonYield, chargeYield, excitonToIonRatio, g1Value, extractionEfficiency, gasGainValue, gasGainWidth, speRes, intrinsicResS1, intrinsicResS2):

	tArgs = [drv.In(seed), drv.In(num_trials), drv.InOut(aS1), drv.InOut(aS2), drv.In(aEnergy), drv.In(photonYield), drv.In(chargeYield), drv.In(excitonToIonRatio), drv.In(g1Value), drv.In(extractionEfficiency), drv.In(gasGainValue), drv.In(gasGainWidth), drv.In(speRes), drv.In(intrinsicResS1), drv.In(intrinsicResS2)]

	observables_func(*tArgs, grid=(512, 1), block=(128, 1, 1))
"""




time.sleep(5)
print 'about to run map'
start_time = time.time()
g_pool.map(f_gpu_observables_func, [tArgs for i in xrange(num_iterations)])
print '\nTime for %d iterations: %.2e s' % (num_iterations, time.time() - start_time)

print aS1

#g_pool.map(f_gpu_practice, [[a_gpu.gpudata], [b_gpu.gpudata], [c_gpu.gpudata], [d_gpu.gpudata]])


#g_pool.map(f_hello, [[1], [2], [3], [4], [5], [6]])
#g_pool.map(f_hello, [[11], [12], [13], [14], [15], [16]])
