import emcee, corner, neriX_analysis
import numpy as np
import collections, sys, os



def smart_stirling(lInput):
	if not hasattr(lInput, '__iter__'):
		lInput = [lInput]
	aOutput = np.array(lInput)
	for index, input in enumerate(lInput):
		if input < 10:
			aOutput[index] = np.log(factorial(input))
		else:
			aOutput[index] = input*np.log(input) - input
	return aOutput



def smart_log(lInput):
	if not hasattr(lInput, '__iter__'):
		lInput = [lInput]
	aOutput = np.array(lInput)
	for index, input in enumerate(lInput):
		if input < 1e-310:
			aOutput[index] = -1000
		else:
			aOutput[index] = np.log(input)
	return aOutput



def smart_log_likelihood_mc(aData, aMC, numMCEvents, confidenceIntervalLimit=0.95):
	totalLogLikelihood = 0.
	for (data, mc) in zip(aData, aMC):
		if mc == 0 and data != 0:
			# use 95% confidence interval
			# confidenceIntervalLimit = 1 - probability of the zero occuring
			probabiltyOfSuccess = 1. - (1.-confidenceIntervalLimit)**(1./numMCEvents)
			totalLogLikelihood += smart_log(smart_binomial(data, numMCEvents, probabiltyOfSuccess))
		else:
			totalLogLikelihood += data*smart_log(mc) - mc - smart_stirling(data)

	return totalLogLikelihood



def smart_log_likelihood(aData, aModel):
	totalLogLikelihood = 0.
	for (data, model) in zip(aData, aModel):
		totalLogLikelihood += data*smart_log(model) - model - smart_stirling(data)
	
	# will be faster to vectorize (if possible)

	return totalLogLikelihood




class mcmc_fitter:

	def __init__(self, fit_name, num_parameters, par_guesses, num_walkers=None, num_steps=None, num_burn_in_steps=None, num_dim=1, num_procs=1, fractional_deviation_guesses=1e-1, par_names=None, save_path=None, num_points=100):

		self.num_parameters = num_parameters
		assert type(self.num_parameters) is int, 'Number of parameters must be an integer!'

		self.par_guesses = tuple(par_guesses)
		assert len(self.par_guesses) == self.num_parameters, 'Number of parameters given does not match the number estimated!'

		if num_walkers == None:
			neriX_analysis.warning_message('Number of walkers not given.  Will use the default of 20 times the number of parameters.')
			self.num_walkers = 20*self.num_parameters

		# save path should be a list where each element
		# is another layer from current directory
		if save_path == None:
			neriX_analysis.warning_message('Save path not given.  Will use the default "./results/".')
			self.save_path = ['results']

		self.fractional_deviation_guesses = fractional_deviation_guesses
		
		assert type(num_points) is int, 'Number of points must be an integer!'
		self.num_points = num_points

		type(num_dim) is int, 'Number of dimensions must be an integer!'
		self.num_dim = num_dim



	# must be called before fitting is started
	def setup_fit(self, function, range_min, range_max, random_state=None):

		self.walkers_start_pos = np.array([(np.random.randn(self.num_parameters)+self.par_guesses)*self.fractional_deviation_guesses + self.par_guesses for i in xrange(self.num_walkers)])

		# if num_dim > 1 range_min and max must be iterable
		if self.num_dim > 1:
			assert isinstance(range_min, collections.Iterable) and isinstance(range_max, collections.Iterable), 'range_min and range_max must be iterable if fitting in more than one dimension.'
		else:
			range_min = [range_min]
			range_max = [range_max]

		assert function.func_code.co_argcount >= self.num_parameters + self.num_dim, 'Function must accept at least the number of parameters plus the number of dimensions!'

		# list of edges will keep the bin edges in each direction
		self.list_of_edges = [0 for i in xrange(self.num_dim)]
		self.list_of_bin_centers = [0 for i in xrange(self.num_dim)]
		self.bin_widths = [0 for i in xrange(self.num_dim)]
		for dim in xrange(self.num_dim):
		
			self.bin_widths[dim] = (range_max[dim] - range_min[dim]) / float(self.num_points)
		
			# need to include +1 in number of samples to account
			# for both edges
			self.list_of_edges[dim] = np.linspace(range_min[dim], range_max[dim], num=self.num_points+1, endpoint=True)
			self.list_of_bin_centers[dim] = np.linspace(range_min[dim] + self.bin_widths[dim]/2., range_max[dim] - self.bin_widths[dim]/2., num=self.num_points, endpoint=True)



	# prior types is a list of tuples that contains the type
	# (uniform or gaussian) and the needed parameters (range
	# or mean/width)
	def fit_data(self, aData, *prior_types):
		pass






if __name__ == '__main__':
	def f_test(a, x):
		pass

	test = mcmc_fitter('test', 1, [10])
	test.setup_fit(f_test, 1, 2)
