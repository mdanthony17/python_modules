#include <vector>
#include <functional>
#include <numeric>
#include "math.h"

// sometimes numpy dot product is not thread safe
// use this in its place if that is the case

/*
float log_likelihood_matching_uncertainty(float data, float scale, int num_mc_events)
{
	data = (int)round(data);
	float ln_likelihood = 0.
	threshold = 50.;
	
	if (num_mc_events < threshold)
	{
		sum_poissons = 0.
		for (int i; i < (threshold + 3*pow(threshold, 0.5)); i++)
		{
			sum_poissons += pow((i/scale), data)*exp(-i/scale)/
		}
		ln_likelihood = log(sum_poissons)
	}
	else
	{
		ln_likelihood = -(data+1)*log(scale) - log(2) - lgamma(data+1) + log(pow((-1),data)-1+ ( 1 + pow(-1, 1+data) + pow(pow(scale-1,2)/pow(scale,2),0.5) + pow(-1, data)*pow(pow(scale-1,2)/pow(scale,2),0.5) )*scale ) + num_mc_events*(1-2*scale)/(2*pow(scale,2)) + (data+1)/2.*log(num_mc_events) + (data-1)/2.*log(num_mc_events*pow((1-scale),2)/pow(scale,2));
	}
	return ln_likelihood;

}
*/



float smart_log_likelihood(float *a_flat_data, float *a_flat_mc, int num_bins, int num_mc_events, float scale_normalized_to_mc_events, float confidence_interval)
{
	float total_log_likelihood = 0.;
	float binom_prob = 1. - pow((1.-confidence_interval),(1./num_mc_events));
	//printf("Beginning: %f\n", total_log_likelihood);
	//printf("lgamma(3): %f\n", lgamma(1));
	//printf("log(3): %f\n", log(3));
	
	for (int bin_number=0; bin_number < num_bins; bin_number++)
	{
		
		//if (a_flat_mc[bin_number]*num_mc_events/scale_normalized_to_mc_events < 800) continue;
		if (a_flat_data[bin_number] < 5) continue;
		
		if (a_flat_data[bin_number] != 0 && a_flat_mc[bin_number] == 0)
		{
			//continue;
			total_log_likelihood += lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob);
			//printf("No mc events: %f\n", lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob));
		}
		else if (a_flat_data[bin_number] == 0 && a_flat_mc[bin_number] == 0)
			continue;
		else
		{
			
			total_log_likelihood += a_flat_data[bin_number]*log(a_flat_mc[bin_number]) - a_flat_mc[bin_number] - lgamma(a_flat_data[bin_number]+1.0);
			//total_log_likelihood += log_likelihood_matching_uncertainty(a_flat_data[bin_number], num_mc_events/scale_normalized_to_mc_events, a_flat_mc[bin_number]*num_mc_events/scale_normalized_to_mc_events);
			//printf("\n\ndata:%f\nmc:%f\nscale:%f\nscaled mc:%f\n", a_flat_data[bin_number], a_flat_mc[bin_number]*num_mc_events/scale_normalized_to_mc_events, num_mc_events/scale_normalized_to_mc_events, a_flat_mc[bin_number]);
			//printf("MC and data (normal likelihood): %f\n", a_flat_data[bin_number]*log(a_flat_mc[bin_number]) - a_flat_mc[bin_number] - lgamma(a_flat_data[bin_number]+1.0));
			//printf("MC and data (matching uncertainty): %f\n", log_likelihood_matching_uncertainty(a_flat_data[bin_number], num_mc_events/scale_normalized_to_mc_events, a_flat_mc[bin_number]*num_mc_events/scale_normalized_to_mc_events));
		}
	}
	//printf("%f", total_log_likelihood);
	return total_log_likelihood;
}