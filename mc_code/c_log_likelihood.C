#include <vector>
#include <functional>
#include <numeric>
#include "math.h"

// sometimes numpy dot product is not thread safe
// use this in its place if that is the case

float smart_log_likelihood(float *a_flat_data, float *a_flat_mc, int num_bins, int num_mc_events, float confidence_interval)
{
	float total_log_likelihood = 0.;
	float binom_prob = 1. - pow((1.-confidence_interval),(1./num_mc_events));
	//printf("Beginning: %f\n", total_log_likelihood);
	//printf("lgamma(3): %f\n", lgamma(1));
	//printf("log(3): %f\n", log(3));
	
	for (int bin_number=0; bin_number < num_bins; bin_number++)
	{
		if (a_flat_data[bin_number] != 0 && a_flat_mc[bin_number] == 0)
		{
			total_log_likelihood += lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob);
			//printf("No mc events: %f\n", lgamma(num_mc_events+1.0) - lgamma(a_flat_data[bin_number]+1.0) - lgamma(num_mc_events-a_flat_data[bin_number]+1.0) + (a_flat_data[bin_number])*log(binom_prob) + (num_mc_events-a_flat_data[bin_number])*log(1-binom_prob));
		}
		else if (a_flat_data[bin_number] == 0 && a_flat_mc[bin_number] == 0)
			continue;
		else
		{
			total_log_likelihood += a_flat_data[bin_number]*log(a_flat_mc[bin_number]) - a_flat_mc[bin_number] - lgamma(a_flat_data[bin_number]+1.0);
			//printf("MC and data: %f\n", log(a_flat_mc[bin_number]));
		}
	}
	//printf("%f", total_log_likelihood);
	return total_log_likelihood;
}