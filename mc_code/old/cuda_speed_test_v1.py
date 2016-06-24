cuda_full_observables_production_code ="""
#include <curand_kernel.h>

extern "C" {

__device__ int gpu_binomial(curandState_t *rand_state, int num_trials, float prob_success)
{
	int x = 0;
	for(int i = 0; i < num_trials; i++) {
		if(curand_uniform(rand_state) < prob_success)
			x += 1;
	}
	return x;
}


__device__ int gpu_poisson_slow(curandState_t *rand_state, float lambda)
{
	int x = 0;
	float p = 1.;
	float exp_neg_lambda = exp(-lambda);
	float u = 0;
	while(p > exp_neg_lambda)
	{
		x += 1;
		u = curand_uniform(rand_state);
		p *= u;
	}
	return x - 1;
}


__device__ int gpu_poisson_fast(curandState_t *rand_state, float lambda)
{
	int c = 0.767 - 3.36/lambda;
	float beta = 3.141592 / pow(3.0*lambda, 0.5);
	float alpha = beta*lambda;
	float k = logf(c) - lambda - logf(beta);
	float u = 0;
	float x = 0;
	int n = 0;
	float v = 0;
	float y = 0;
	float lhs = 0.;
	float rhs = 0.;
	
	while(1)
	{
		u = curand_uniform(rand_state);
		x = (alpha - logf((1.0 - u)/u))/beta;
		n = floor(x + 0.5);
		if(n < 0) continue;
		
		u = curand_uniform(rand_state);
		y = alpha - beta*x;
		lhs = y + logf(v/pow(1.0 + exp(y),2));
		rhs = k + n*logf(lambda) - n*logf(n) + n;
		if (lhs <= rhs) return n;
	}
}


__global__ void gpu_full_observables_production(int *seed, int *num_trials, float *aS1, float *aS2, float *aEnergy, float *photonYield, float *chargeYield, float *excitonToIonRatio, float *g1Value, float *extractionEfficiency, float *gasGainValue, float *gasGainWidth, float *speRes, float *intrinsicResS1, float *intrinsicResS2)
{

	// start random number generator
	curandState_t s;
	//const int iteration = blockIdx.x * blockDim.x + threadIdx.x;
	const int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	//curand_init(0, 0, 0, &s); // for debugging
	curand_init(*seed * iteration, 0, 0, &s);
	
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
		
		
		//mcQuanta = curand_poisson(&s, (double)mcEnergy*(*photonYield + *chargeYield));
		//mcQuanta = gpu_poisson_slow(&s, mcEnergy*(*photonYield + *chargeYield));
		mcQuanta = gpu_poisson_fast(&s, mcEnergy*(*photonYield + *chargeYield));
		//mcQuanta = (curand_normal(&s)*pow(mcEnergy*(*photonYield + *chargeYield), 0.5)) + mcEnergy*(*photonYield + *chargeYield);
		//aS1[iteration] = mcQuanta;
		return;
		
		
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




import sys, os, random, time
import numpy as np

import cuda_full_observables_production
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.tools
import pycuda.gpuarray

import rootpy.compiled as C
C.register_file('c_full_observables_production_no_eff.C', ['full_matching_loop'])
c_full_matching_loop = C.full_matching_loop


drv.init()
dev = drv.Device(0)
ctx = dev.make_context(drv.ctx_flags.SCHED_AUTO | drv.ctx_flags.MAP_HOST)

grid_d1 = 512
block_d1 = 1024
num_entries = int(grid_d1*block_d1)
num_iterations = 100

"""
practie_kernel = pycuda.compiler.SourceModule(\"""
__global__ void krnl(float *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = a[i]+1;
}
\""").get_function("krnl")

a = drv.pagelocked_zeros((num_entries, 1), np.float32, mem_flags=drv.host_alloc_flags.DEVICEMAP)

aa = np.intp(a.base.get_device_pointer())

# line to run gpu code
startTime = time.time()
practie_kernel(aa, grid=(num_entries,1), block=(1,1,1))
print 'Initial time: %f\n' % (time.time() - startTime)

startTime = time.time()
practie_kernel(aa, grid=(num_entries,1), block=(1,1,1))
print 'Second time: %f\n' % (time.time() - startTime)

#print list(a)
"""

#a = drv.pagelocked_zeros((num_entries, 1), np.float32, mem_flags=drv.host_alloc_flags.DEVICEMAP)
#a = np.zeros(num_entries, dtype=np.float32)
#aEnergy = drv.register_host_memory(a)
#aEnergy = drv.mem_alloc(a.nbytes)
#a_pin = drv.pagelocked_empty(shape=num_entries, dtype=np.float32)
#a_pin.fill(10.)


observables_func = pycuda.compiler.SourceModule(cuda_full_observables_production_code, no_extern_c=True).get_function('gpu_full_observables_production')



# set variables for full matching

aEnergy = np.full(num_entries, 10., dtype=np.float32)
aEnergy_gpu = pycuda.gpuarray.to_gpu(aEnergy)

aS1 = np.full(num_entries, -1, dtype=np.float32)
aS1_gpu = pycuda.gpuarray.to_gpu(aS1)
aS2 = np.full(num_entries, -1, dtype=np.float32)
aS2_gpu = pycuda.gpuarray.to_gpu(aS2)

seed = np.asarray(int(time.time()*1000), dtype=np.int32)
num_trials = np.asarray(num_entries, dtype=np.int32)
photonYield = np.asarray(5, dtype=np.float32)
chargeYield = np.asarray(5, dtype=np.float32)
excitonToIonRatio = np.asarray(.5, dtype=np.float32)
g1Value = np.asarray(.1, dtype=np.float32)
extractionEfficiency = np.asarray(.9, dtype=np.float32)
gasGainValue = np.asarray(13, dtype=np.float32)
gasGainWidth = np.asarray(5., dtype=np.float32)
speRes = np.asarray(.7, dtype=np.float32)
intrinsicResS1 = np.asarray(.3, dtype=np.float32)
intrinsicResS2 = np.asarray(.1, dtype=np.float32)
tacEff = np.asarray([1e6], dtype=np.float32)
pfEff = np.asarray([-1e6, 1], dtype=np.float32)
trigEff = np.asarray([-1e6, 1], dtype=np.float32)

tArgs = [drv.In(seed), drv.In(num_trials), aS1_gpu.gpudata, aS2_gpu.gpudata, aEnergy_gpu.gpudata, drv.In(photonYield), drv.In(chargeYield), drv.In(excitonToIonRatio), drv.In(g1Value), drv.In(extractionEfficiency), drv.In(gasGainValue), drv.In(gasGainWidth), drv.In(speRes), drv.In(intrinsicResS1), drv.In(intrinsicResS2)]

observables_func(*tArgs, grid=(grid_d1,1), block=(block_d1,1,1))

print 'Start GPU operation.'
startTime = time.time()
for i in xrange(num_iterations):
	observables_func(*tArgs, grid=(grid_d1,1), block=(block_d1,1,1))
	aS1 = aS1_gpu.get_async()
	aS2 = aS2_gpu.get_async()
	#seed = np.asarray(int(time.time()*1000), dtype=np.int32)
	#tArgs[0] = drv.In(seed)
	print aS1
	print aS2

print 'Time for %d iterations on GPU: %f\n' % (num_iterations, time.time() - startTime)

ctx.pop()


s_test_function = """
#include <math.h>

void test_function(int *numTrials, float *aValues)
{
	for (int i=0; i < *numTrials; i++)
	{
		aValues[i] = aValues[i] + 1;
	}
}


"""

"""
C.register_code(s_test_function, ['test_function'])
c_test_function = C.test_function

aValues = np.zeros(num_entries, dtype=np.float32)

num_trials = np.asarray(num_entries, dtype=np.int32)

startTime = time.time()
c_test_function(num_trials, aValues)
print 'C loop time: %f\n' % (time.time() - startTime)
"""

"""
aEnergy = np.full(num_entries, 10, dtype=np.float32)
aS1 = np.full(num_entries, -1, dtype=np.float32)
aS2 = np.full(num_entries, -1, dtype=np.float32)

print 'Start CPU operation.'
startTime = time.time()
for i in xrange(num_iterations):
	c_full_matching_loop(seed, num_trials, aS1, aS2, aEnergy, photonYield, chargeYield, excitonToIonRatio, g1Value, extractionEfficiency, gasGainValue, gasGainWidth, speRes, intrinsicResS1, intrinsicResS2, tacEff, trigEff, pfEff)
print 'Time for %d iterations on CPU: %f\n' % (num_iterations, time.time() - startTime)
"""
