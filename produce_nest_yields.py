import ROOT as root
from math import exp, log, floor
import sys

def nest_nr_mean_yields(keVNR, dfVcm):
	# constants
	rho = 2.9 # g/cc
	tiParameter = 3.77e-2
	atomicNumber = 131
	
	epsilon = 11.5*keVNR*((54.)**(-7./3.))
	kFactor = 0.1394*(atomicNumber/131.293)**0.5
	g = 3.*(epsilon**(0.15))+0.7*(epsilon**(0.6))+epsilon
	lindhardFactor = (kFactor*g)/(1.+kFactor*g)
	numQuanta = keVNR*lindhardFactor*1000./(21.717-2.7759*rho)
	excitonToIonRatio = 1.24*dfVcm**-0.0472*(1.-exp(-239.*epsilon))
	numIons = numQuanta/(1.+excitonToIonRatio)
	recombinationProb = 1.-log(1.+(tiParameter/4.)*numIons)/((tiParameter/4.)*numIons)

	numPhotons = (numQuanta*(excitonToIonRatio/(1.+excitonToIonRatio))+numIons*recombinationProb)/(1+3.32*epsilon**1.141)
	numElectrons = (1.-recombinationProb)*numIons

	return (numPhotons/float(keVNR), numElectrons/float(keVNR))



#def nest_single_nr_mc(alpha, zeta, beta, gamma, delta, k_factor, eta, l_factor, c_factor):
	# follow naming convention of NEST parameterization
	# with the exception of k, lambda, and C because the names needed
	# to change





def nest_er_yields(keVER, dfVcm):
	# using 1.0 rootNest.cc file
	# code looks like it was written by 3 year old
	# translating variables is a PAIN
	# ke = recoil energy
	# alf = 1./(1+exciton_ion_ratio) = prob_ion
	
	rho = 2.9 # g/cc
	w_value = (21.717-2.7759*rho)*1e-3
	log10_df = log(dfVcm, 10.)
	
	# get exciton to ion ratio
	if keVER < 1.3:
		exciton_ion_ratio = 0.06
	else:
		exciton_ion_ratio = (0.059709 + 0.048577*rho)*(1. - exp(-1.5*(keVER-0.19)))

	prob_ion = 1. / (1.+exciton_ion_ratio)
	
	

	# find all TIB related quantities
	tib = 0.10077*pow(dfVcm, -0.078314)

	tib_m1 = 0.0024389+((0.50042-0.0024389)/(1.+(dfVcm/11.747)**0.64723))
	tib_m2 = 2.6927000+((160.500-2.6927000)/(1.+(dfVcm/28.518)**0.95537))
	tib_m3 = 398.08000+((23223.0-398.08000)/(1.+(dfVcm/35.008)**1.06550))
	tib_m4 = 54083.000+((3.2519e6-54083.00)/(1.+(dfVcm/37.286)**1.11110))
	tib_m5 = 2.35600e6+((1.3762e8-2.3560e6)/(1.+(dfVcm/36.118)**1.11000))

	tib_low = 0.045*dfVcm**-0.13 * (1.-exp(-0.035*keVER**4.-0.13))
	tib_high = tib_m1*keVER**-1. + tib_m2*keVER**-2. - tib_m3*keVER**-3. + tib_m4*keVER**-4. - tib_m5*keVER**-5.
	tib_100 = tib_m1/100. + tib_m2/(100.**2.) - tib_m3/(100.**3.) + tib_m4/(100.**4.) - tib_m5/(100.**5.)



	# want recombination probability
	if keVER > 18.6 and keVER < 122.:
		
		pol0_py = 57.92384-8.097075*log10_df + 12.31535*log10_df**2. - 15.4544*log10_df**3. + 8.911858*log10_df**4. - 2.675756*log10_df**5. + 0.3966776*log10_df**6. - 0.02281601*log10_df**7.
		pol1_py = 0.1338665 - 0.413536*log10_df + 1.836502*log10_df**2. - 3.052137*log10_df**3. + 2.780416*log10_df**4. - 1.511113*log10_df**5. + 0.4935669*log10_df**6. - 0.09448295*log10_df**7. + 0.009755602*log10_df**8. - 0.000419436*log10_df**9.
		pol2_py = -0.0007282148 + 0.003062095*log10_df - 0.01255166*log10_df**2. + 0.02011632*log10_df**3. - 0.0177259*log10_df**4. + 0.009324331*log10_df**5. - 0.00296341*log10_df**6. + 0.0005555244*log10_df**7. - 5.647981e-5*log10_df**8. + 2.401166e-6*log10_df**9.

		photon_yield = pol0_py + pol1_py*keVER + pol2_py*keVER**2.
		electron_yield = 1./w_value - photon_yield
		#return photon_yield, electron_yield

		mean_num_ions = floor(keVER/w_value + 0.5)*prob_ion
		mean_num_excitons = floor(keVER/w_value + 0.5) - mean_num_ions
		prob_recombination = (photon_yield*keVER - mean_num_excitons) / mean_num_ions


	# want recombination probability
	elif keVER < 1.3:
	
		prob_ion = 1./1.019
	
		pol0_ey = 77.111 + ( 76.329 - 77.111 ) / ( 1. + pow((log10_df/3.3739),2.6411) )
		pol1_ey = 21.698 + ( 14.215 - 21.698 ) / ( 1. + pow((log10_df/4.7893),1.9549) )
		pol2_ey = 15.821 + (-.55771 - 15.821 ) / ( 1. + pow((log10_df/10.739),1.1941) )
	
		electron_yield = pol0_ey - pol1_ey*keVER + pol2_ey*keVER**2.
		if electron_yield > 73.:
			electron_yield = 73.

		temp_num_ions = floor(keVER/w_value+0.5)*prob_ion
		prob_recombination = 1. - ((electron_yield*keVER) / temp_num_ions)

		#photon_yield = 1./w_value - electron_yield
		#return photon_yield, electron_yield

	# want TIB
	elif keVER >= 1.3 and keVER <= 18.6:
		e_curv = 45.836*(dfVcm**-0.203)
		tib = tib_100+((tib_low*1.1-tib_100)/(1.+((keVER/e_curv)**2.0)))

	else:
		tib = tib_high


	mean_num_quanta = floor(keVER/w_value + 0.5)
	mean_num_ions = mean_num_quanta*prob_ion
	
	#print prob_ion, exciton_ion_ratio
	
	if (keVER >= 1.3 and keVER <= 18.6) or keVER > 122.:
		tib *= (rho/2.888)**0.3
		prob_recombination = 1. - log(1.+(mean_num_ions/4.)*tib) / ((mean_num_ions/4.)*tib)

	#print tib
	
	if prob_recombination > 1:
		prob_recombination = 1
	elif prob_recombination < 0:
		prob_recombination = 0

	#print 'recombination: %f' % prob_recombination

	eff_mean_num_ions = mean_num_ions
	if eff_mean_num_ions > 3e4:
		eff_mean_num_ions = 3e4

	fr_const = 0.0075 * eff_mean_num_ions
	if fr_const < 0:
		fr_const = 0

	if keVER > 12259.*dfVcm**-1.218 and keVER < 100.:
		fr_const = 3.**0.5 * 0.0075 * eff_mean_num_ions

	#print 'fr_const: %f' % fr_const

	if fr_const <= 0:
		num_electrons = floor(mean_num_ions*(1.-prob_recombination) + 0.5)
	else:
		man = 0.85607 + 0.032441 * log10_df
		pwr = .023162 - .0053890 * log10_df
		kludge = man * keVER**pwr
		#print kludge
		if kludge < 1.:
			kludge = 1.
		mean_num_electrons = kludge*mean_num_ions*(1.-prob_recombination)

	if mean_num_electrons > mean_num_ions:
		mean_num_electrons = mean_num_ions

	mean_num_photons = mean_num_quanta - mean_num_electrons

	return mean_num_photons/keVER, mean_num_electrons/keVER


if __name__ == '__main__':
	#print nest_nr_yields(2, 500)
	print nest_er_yields(40, 1000)

