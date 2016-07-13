
# example using pagelocked memory (pinned)

"""
##CODE START######################
import pycuda.driver as drv

drv.init()
dev = drv.Device(0)
ctx = dev.make_context(drv.ctx_flags.SCHED_AUTO | drv.ctx_flags.MAP_HOST)

k = pycuda.compiler.SourceModule(\"""
__global__ void krnl(float* a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = i;
}
\""").get_function("krnl")

a = drv.pagelocked_empty((10, 10), numpy.float32, mem_flags=drv.host_alloc_flags.DEVICEMAP)

aa = numpy.intp(a.base.get_device_pointer())
k(aa, grid=(100,1), block=(1,1,1))

ctx.pop()


##CODE END######################
"""






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

// used for finding index for 2d histogram array
// lower bound corresponds to the index
// uses binary search ON SORTED ARRAY
// THIS IS THE TEST WHICH MUST RETURN VOIDS
// AND HAVE POINTER INPUTS
__global__ void test_gpu_find_lower_bound(int *num_elements, float *a_sorted, float *search_value, int *index)
{
	float *first = a_sorted;
	float *iterator = a_sorted;
	int count = *num_elements;
	int step;
	
	if (*search_value < a_sorted[0] || *search_value > a_sorted[*num_elements])
	{
		*index = -1;
		return;
	}
	
	while (count > 0)
	{
		iterator = first;
		step = count / 2;
		iterator += step;
		if (*iterator < *search_value)
		{
			first = ++iterator;
			count -= step + 1;
		}
		else
		{
			count = step;
		}
		// -1 to get lower bound
		*index = iterator - a_sorted - 1;
	}

}


// used for finding index for 2d histogram array
// lower bound corresponds to the index
// uses binary search ON SORTED ARRAY
__device__ int gpu_find_lower_bound(int *num_elements, float *a_sorted, float search_value)
{
	float *first = a_sorted;
	float *iterator = a_sorted;
	int count = *num_elements;
	int step;
	
	if (search_value < a_sorted[0] || search_value > a_sorted[*num_elements])
	{
		return -1;
	}
	
	while (count > 0)
	{
		iterator = first;
		step = count / 2;
		iterator += step;
		if (*iterator < search_value)
		{
			first = ++iterator;
			count -= step + 1;
		}
		else
		{
			count = step;
		}
	}
	// -1 to get lower bound
	return iterator - a_sorted - 1;

}



__global__ void gpu_full_observables_production_with_hist(int *seed, int *num_trials, float *aEnergy, float *photonYield, float *chargeYield, float *excitonToIonRatio, float *g1Value, float *extractionEfficiency, float *gasGainValue, float *gasGainWidth, float *speRes, float *intrinsicResS1, float *intrinsicResS2, float *tacEff, float *trigEff, float *pfEff, float *nr_band_cut, int *num_bins_s1, float *bin_edges_s1, int *num_bins_s2, float *bin_edges_s2, int *hist_2d_array)
{

	// start random number generator
	curandState s;
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
			return;
		}
		if (mcElectrons < 1 || *extractionEfficiency < 0 || *extractionEfficiency > 1) 
		{	
			return;
		}
		if (*gasGainWidth <= 0) 
		{	
			return;
		}
		
		mcS1 = gpu_binomial(&s, mcPhotons, *g1Value);
		mcExtractedElectrons = gpu_binomial(&s, mcElectrons, *extractionEfficiency);
		mcS2 = (curand_normal(&s) * *gasGainWidth*powf(mcExtractedElectrons, 0.5)) + mcExtractedElectrons**gasGainValue;
		
		if (mcS1 < 0) 
		{	
			return;
		}
		if (mcS2 < 0) 
		{	
			return;
		}
		
		//aS1[iteration] = mcS1;
		//aS2[iteration] = mcS2;
		//return;
		
		
		
		// Band cut
		if (mcS2 > nr_band_cut[0] + nr_band_cut[1]*mcS1 + nr_band_cut[2]*mcS1*mcS1)
		{	
			return;
		}

		
		
		// ------------------------------------------------
		//  Smear S1 and S2
		// ------------------------------------------------
		
		if (*speRes <= 0 || *intrinsicResS1 <= 0 || *intrinsicResS2 <= 0) 
		{	
			return;
		}
		
		mcS1 = (curand_normal(&s) * *speRes*powf(mcS1, 0.5)) + mcS1;
		if (mcS1 < 0) 
		{	
			return;
		}
		mcS1 = (curand_normal(&s) * *intrinsicResS1*mcS1) + mcS1;
		if (mcS1 < 0) 
		{	
			return;
		}
		
		
		mcS2 = (curand_normal(&s) * *intrinsicResS2*mcS2) + mcS2;
		if (mcS2 < 0) 
		{	
			return;
		}
		
		//aS1[iteration] = mcS1;
		//aS2[iteration] = mcS2;
		
		
		// tof_efficiency
		if (curand_uniform(&s) > (1. - exp(-tacEff[0] * mcS1)))
		{	
			return;
		}
		
		// trig efficiency
		if (curand_uniform(&s) > 1. / (1 + exp(-(mcS2-trigEff[0])/trigEff[1])))
		{	
			return;
		}
		
		// peak finder efficiency
		if (curand_uniform(&s) > (1. - exp(-(mcS1-pfEff[0])/pfEff[1])))
		{	
			return;
		}
		
		// aS1[iteration] = mcS1;
		// aS2[iteration] = mcS2;
		
		// find indices of s1 and s2 bins for 2d histogram
		
		int s1_bin = gpu_find_lower_bound(num_bins_s1, bin_edges_s1, mcS1);
		int s2_bin = gpu_find_lower_bound(num_bins_s2, bin_edges_s2, mcS2);
		
		
		if (s1_bin == -1 || s2_bin == -1)
		{
			return;
		}
		
		// hist_2d_array[0] = s1_bin + *num_bins_s1*s2_bin;
		// hist_2d_array[1] = s1_bin;
		// hist_2d_array[2] = s2_bin;
		
		hist_2d_array[s1_bin + *num_bins_s1*s2_bin] += 1;
		
		return;
	
	}

  
}




__global__ void gpu_full_observables_production_with_hist_spline(int *seed, int *num_trials, float *meanField, float *aEnergy, int *numSplinePoints, float *aEnergySplinePoints, float *aPhotonYieldSplinePoints, float *aChargeYieldSplinePoints, float *g1Value, float *extractionEfficiency, float *gasGainValue, float *gasGainWidth, float *speRes, float *intrinsicResS1, float *intrinsicResS2, float *excitonToIonPar0RV, float *excitonToIonPar1RV, float *excitonToIonPar2RV, float *s1_eff_par0, float *s1_eff_par1, float *s2_eff_par0, float *s2_eff_par1, float *nr_band_cut, int *num_bins_s1, float *bin_edges_s1, int *num_bins_s2, float *bin_edges_s2, int *hist_2d_array)
{

	// start random number generator
	curandState s;
	//const int iteration = blockIdx.x * blockDim.x + threadIdx.x;
	const int iteration = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	//curand_init(0, 0, 0, &s); // for debugging
	curand_init(*seed * iteration, 0, 0, &s);
	
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
	float slopeForPhotonYield;
	float slopeForChargeYield;
	
	float excitonToIonRatio;
	float excitonToIonPar0;
	float excitonToIonPar1;
	float excitonToIonPar2;
	
	float photonYield;
	float chargeYield;
	int indexOfUpperSplinePoint;
	
	float probRecombination;
	
	if (iteration < *num_trials)
	{
	
		// ------------------------------------------------
		//  Draw random energy from distribution
		// ------------------------------------------------
		
		
		mcEnergy = aEnergy[iteration];
		//aS1[iteration] = mcEnergy;
		//return;
		
		if (mcEnergy < aEnergySplinePoints[0] || mcEnergy > aEnergySplinePoints[*numSplinePoints-1]) return;
		
		
		// ------------------------------------------------
		//  Interpolate the photon and charge yield
		// ------------------------------------------------

		indexOfUpperSplinePoint = 1;
		
		while (aEnergySplinePoints[indexOfUpperSplinePoint] < mcEnergy)
			indexOfUpperSplinePoint += 1;
		
		slopeForPhotonYield = (aPhotonYieldSplinePoints[indexOfUpperSplinePoint]-aPhotonYieldSplinePoints[indexOfUpperSplinePoint-1]) / (aEnergySplinePoints[indexOfUpperSplinePoint]-aEnergySplinePoints[indexOfUpperSplinePoint-1]);
		slopeForChargeYield = (aChargeYieldSplinePoints[indexOfUpperSplinePoint]-aChargeYieldSplinePoints[indexOfUpperSplinePoint-1]) / (aEnergySplinePoints[indexOfUpperSplinePoint]-aEnergySplinePoints[indexOfUpperSplinePoint-1]);

		photonYield = aPhotonYieldSplinePoints[indexOfUpperSplinePoint] + slopeForPhotonYield*(mcEnergy - aEnergySplinePoints[indexOfUpperSplinePoint]);
		chargeYield = aChargeYieldSplinePoints[indexOfUpperSplinePoint] + slopeForChargeYield*(mcEnergy - aEnergySplinePoints[indexOfUpperSplinePoint]);
		
		// if (mcEnergy > 10 && mcEnergy < 11)
		// {
		// 	hist_2d_array[0] = indexOfUpperSplinePoint;
		// 	hist_2d_array[1] = aEnergySplinePoints[indexOfUpperSplinePoint];
		// 	return;
		// }
		// else return;
		
		

		// ------------------------------------------------
		//  Find number of quanta
		// ------------------------------------------------
		
		
		mcQuanta = curand_poisson(&s, mcEnergy*(photonYield + chargeYield));
		//aS1[iteration] = mcQuanta;
		//return;
		
		
		
		// ------------------------------------------------
		//  Calculate exciton to ion ratio
		// ------------------------------------------------
		
		if (*excitonToIonPar0RV > 0)
			excitonToIonPar0 = 1.240 + *excitonToIonPar0RV*0.079;
		else
			excitonToIonPar0 = 1.240 - *excitonToIonPar0RV*0.073;
		
		if (*excitonToIonPar1RV > 0)
			excitonToIonPar1 = 0.0472 + *excitonToIonPar1RV*0.0088;
		else
			excitonToIonPar1 = 0.0472 - *excitonToIonPar1RV*0.0073;
		
		if (*excitonToIonPar2RV > 0)
			excitonToIonPar2 = 239.0 + *excitonToIonPar2RV*28.0;
		else
			excitonToIonPar2 = 239.0 - *excitonToIonPar2RV*8.8;
		
		excitonToIonRatio = excitonToIonPar0*powf(*meanField,-excitonToIonPar1) * ( 1 - exp(-excitonToIonPar2 * 11.5*mcEnergy*powf(54, -7./3.)) );
		
		probRecombination = ( (excitonToIonRatio+1) * photonYield )/(photonYield+chargeYield) - excitonToIonRatio;
		
		
		
		// ------------------------------------------------
		//  Convert to excitons and ions
		// ------------------------------------------------
		
		
		probExcitonSuccess = 1. - 1./(1. + excitonToIonRatio);
		if (probExcitonSuccess < 0 || probExcitonSuccess > 1) 
		{	
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
			return;
		}
		if (mcElectrons < 1 || *extractionEfficiency < 0 || *extractionEfficiency > 1) 
		{	
			return;
		}
		if (*gasGainWidth <= 0) 
		{	
			return;
		}
		
		mcS1 = gpu_binomial(&s, mcPhotons, *g1Value);
		mcExtractedElectrons = gpu_binomial(&s, mcElectrons, *extractionEfficiency);
		mcS2 = (curand_normal(&s) * *gasGainWidth*powf(mcExtractedElectrons, 0.5)) + mcExtractedElectrons**gasGainValue;
		
		if (mcS1 < 0) 
		{	
			return;
		}
		if (mcS2 < 0) 
		{	
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
			return;
		}
		
		mcS1 = (curand_normal(&s) * *speRes*powf(mcS1, 0.5)) + mcS1;
		if (mcS1 < 0) 
		{	
			return;
		}
		mcS1 = (curand_normal(&s) * *intrinsicResS1*mcS1) + mcS1;
		if (mcS1 < 0) 
		{	
			return;
		}
		
		
		mcS2 = (curand_normal(&s) * *intrinsicResS2*mcS2) + mcS2;
		if (mcS2 < 0) 
		{	
			return;
		}
		
		
				
		
		// Band cut
		if (mcS1 < 24 && (mcS2 > nr_band_cut[0] + nr_band_cut[1]*mcS1 + nr_band_cut[2]*mcS1*mcS1))
		{	
			return;
		}
		
	
		
		// trig efficiency
		if (curand_uniform(&s) > 1. / (1 + exp(-(mcS2-*s2_eff_par0) / *s2_eff_par1)))
		{	
			return;
		}
		
		// peak finder efficiency
		// if (curand_uniform(&s) > (1 - exp(-(mcS1-*s1_eff_par0) / *s1_eff_par1)))
		if (curand_uniform(&s) > (exp(-*s1_eff_par0*exp(-mcS1 * *s1_eff_par1))))
		{
			return;
		}
		
			
		// hist_2d_array[0] = num_bins_s1;
		// hist_2d_array[1] = num_bins_s2;
		// return;
		
		// find indices of s1 and s2 bins for 2d histogram
		
		int s1_bin = gpu_find_lower_bound(num_bins_s1, bin_edges_s1, mcS1);
		int s2_bin = gpu_find_lower_bound(num_bins_s2, bin_edges_s2, mcS2);
		
		
		if (s1_bin == -1 || s2_bin == -1)
		{
			return;
		}
		
		// hist_2d_array[0] = s1_bin + *num_bins_s1*s2_bin;
		// hist_2d_array[1] = s1_bin;
		// hist_2d_array[2] = s2_bin;
		
		hist_2d_array[s1_bin + *num_bins_s1*s2_bin] += 1;
		
		return;
	
	}

  
}






__global__ void gpu_full_observables_production(int *seed, int *num_trials, float *aS1, float *aS2, float *aEnergy, float *photonYield, float *chargeYield, float *excitonToIonRatio, float *g1Value, float *extractionEfficiency, float *gasGainValue, float *gasGainWidth, float *speRes, float *intrinsicResS1, float *intrinsicResS2, float *tacEff, float *trigEff, float *pfEff)
{

	// start random number generator
	curandState s;
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
		
		//aS1[iteration] = mcS1;
		//aS2[iteration] = mcS2;
		
		
		// tof_efficiency
		if (curand_uniform(&s) > (1. - exp(-tacEff[0] * mcS1)))
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		// trig efficiency
		if (curand_uniform(&s) > 1. / (1 + exp(-(mcS2-trigEff[0])/trigEff[1])))
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		// peak finder efficiency
		if (curand_uniform(&s) > (1. - exp(-(mcS1-pfEff[0])/pfEff[1])))
		{	
			aS1[iteration] = -1;
			aS2[iteration] = -1;
			return;
		}
		
		aS1[iteration] = mcS1;
		aS2[iteration] = mcS2;
		return;
	
	}

  
}

}
"""