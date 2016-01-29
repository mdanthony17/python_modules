import ROOT as root
from math import exp, log

def nest_nr_yields(keVNR, dfVcm):
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






if __name__ == '__main__':
	print nest_nr_yields(2, 500)

