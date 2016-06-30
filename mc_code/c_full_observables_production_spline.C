#include <vector>
#include <TH1.h>
#include <TRandom3.h>
#include <stdio.h>
#include <math.h>

void full_matching_loop(int *seed, int *numTrials, float *meanField, float *aS1, float *aS2, float *aEnergy, int *numSplinePoints, float *aEnergySplinePoints, float *aPhotonYieldSplinePoints, float *aChargeYieldSplinePoints, float *g1Value, float *speRes, float *extractionEfficiency, float *gasGainValue, float *gasGainWidth, float *intrinsicResS1, float *intrinsicResS2, float *excitonToIonPar0RV, float *excitonToIonPar1RV, float *excitonToIonPar2RV, float *s1_eff_par0, float *s1_eff_par1, float *s2_eff_par0, float *s2_eff_par1)
{
	TRandom r3 = TRandom3(*seed);
	
	// variables needed in loop
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

	for (int i = 0; i < *numTrials; i++)
	{
		// ------------------------------------------------
		//  Draw random energy from distribution
		// ------------------------------------------------
		
		mcEnergy = aEnergy[i];
		//printf("Energy: %f \n", mcEnergy);

		// don't both if photon and charge yield undefined/too small
		// or energy is too large (low stats)
		if (mcEnergy < aEnergySplinePoints[0] || mcEnergy > aEnergySplinePoints[*numSplinePoints-1]) continue;

		// ------------------------------------------------
		//  Interpolate the photon and charge yield
		// ------------------------------------------------

		indexOfUpperSplinePoint = 1;
		//printf("%f \n", aEnergySplinePoints[indexOfUpperSplinePoint]);
		
		while (aEnergySplinePoints[indexOfUpperSplinePoint] < mcEnergy)
			indexOfUpperSplinePoint++;
		
		slopeForPhotonYield = (aPhotonYieldSplinePoints[indexOfUpperSplinePoint]-aPhotonYieldSplinePoints[indexOfUpperSplinePoint-1]) / (aEnergySplinePoints[indexOfUpperSplinePoint]-aEnergySplinePoints[indexOfUpperSplinePoint-1]);
		//printf("%f \n", slopeForPhotonYield);
		slopeForChargeYield = (aChargeYieldSplinePoints[indexOfUpperSplinePoint]-aChargeYieldSplinePoints[indexOfUpperSplinePoint-1]) / (aEnergySplinePoints[indexOfUpperSplinePoint]-aEnergySplinePoints[indexOfUpperSplinePoint-1]);
		//printf("%f \n", slopeForChargeYield);

		photonYield = aPhotonYieldSplinePoints[indexOfUpperSplinePoint] + slopeForPhotonYield*(mcEnergy - aEnergySplinePoints[indexOfUpperSplinePoint]);
		//printf("%f \n", photonYield);
		chargeYield = aChargeYieldSplinePoints[indexOfUpperSplinePoint] + slopeForChargeYield*(mcEnergy - aEnergySplinePoints[indexOfUpperSplinePoint]);
		//printf("%f \n", chargeYield);

		//printf("%f \n", aEnergySplinePoints[indexOfUpperSplinePoint]);


		// ------------------------------------------------
		//  Find number of quanta
		// ------------------------------------------------
		
		
		mcQuanta = r3.Poisson(mcEnergy*(photonYield + chargeYield));
		//printf("%d \\n", numQuanta);
		
		
		
		// ------------------------------------------------
		//  Calculate exciton to ion ratio
		// ------------------------------------------------
		
		if (excitonToIonPar0RV > 0)
			excitonToIonPar0 = 1.240 + *excitonToIonPar0RV*0.079;
		else
			excitonToIonPar0 = 1.240 - *excitonToIonPar0RV*0.073;
		
		if (excitonToIonPar1RV > 0)
			excitonToIonPar1 = 0.0472 + *excitonToIonPar1RV*0.0088;
		else
			excitonToIonPar1 = 0.0472 - *excitonToIonPar1RV*0.0073;
		
		if (excitonToIonPar2RV > 0)
			excitonToIonPar2 = 239.0 + *excitonToIonPar2RV*28.0;
		else
			excitonToIonPar2 = 239.0 - *excitonToIonPar2RV*8.8;
		
		excitonToIonRatio = excitonToIonPar0*pow(*meanField,-excitonToIonPar1) * ( 1 - exp(-excitonToIonPar2 * 11.5*mcEnergy*pow(54, -7./3.)) );
		//printf("%f \n", excitonToIonRatio);
		
		probRecombination = ( (excitonToIonRatio+1) * photonYield )/(photonYield+chargeYield) - excitonToIonRatio;
		
		// ------------------------------------------------
		//  Convert to excitons and ions
		// ------------------------------------------------
		
		
		probExcitonSuccess = 1. - 1./(1. + excitonToIonRatio);
		if (probExcitonSuccess < 0 || probExcitonSuccess > 1) continue;
		
		mcExcitons = r3.Binomial(mcQuanta, probExcitonSuccess);
		mcIons = mcQuanta - mcExcitons;
		//printf("%d \\n", mcExcitons);
		
		
		// ------------------------------------------------
		//  Ion recombination
		// ------------------------------------------------

		if (mcIons < 1 || probRecombination < 0 || probRecombination > 1) continue;
		
		mcRecombined = r3.Binomial(mcIons, probRecombination);
		mcPhotons = mcExcitons + mcRecombined;
		mcElectrons = mcIons - mcRecombined;
		//printf("%d \\n", mcRecombined);
		
		
		// ------------------------------------------------
		//  Convert to S1 and S2 BEFORE smearing
		// ------------------------------------------------
		
		if (mcPhotons < 1 || *g1Value < 0 || *g1Value > 1) continue;
		if (mcElectrons < 1 || *extractionEfficiency < 0 || *extractionEfficiency > 1) continue;
		if (*gasGainWidth <= 0) continue;
		
		mcS1 = r3.Binomial(mcPhotons, *g1Value);
		
		//printf("DPE taken into account but hardcoded!");
		//mcS1 = r3.Binomial(mcPhotons, *g1Value / 1.21);
		//float dpe = r3.Binomial(mcS1, 0.21);
		//mcS1 = mcS1 + dpe;
		
		
		mcExtractedElectrons = r3.Binomial(mcElectrons, *extractionEfficiency);
		mcS2 = r3.Gaus(mcExtractedElectrons * *gasGainValue, *gasGainWidth*pow(mcExtractedElectrons, 0.5));
		
		if (mcS1 <= 0) continue;
		if (mcS2 <= 0) continue;
		
		//printf("%f \\n", mcS1);
		//printf("%f \\n", mcS2);
		
		
		// ------------------------------------------------
		//  Smear S1 and S2
		// ------------------------------------------------
		
		if (*speRes <= 0 || *intrinsicResS1 <= 0 || *intrinsicResS2 <= 0) continue;
		
		mcS1 = r3.Gaus(mcS1, *speRes*pow(mcS1, 0.5));
		if (mcS1 <= 0) continue;
		mcS1 = r3.Gaus(mcS1, *intrinsicResS1*mcS1);
		if (mcS1 <= 0) continue;
		
		// no SPE smearing (included in gas gain)
		//mcS2 = r3.Gaus(mcS2, *speRes*pow(mcS2, 0.5));
		//if (mcS2 < 0) continue;
		mcS2 = r3.Gaus(mcS2, *intrinsicResS2*mcS2);
		if (mcS2 <= 0) continue;
		
		// s2 efficiency
		//printf("S2: %f \n", mcS2);
		//printf("eff: %f \n", 1. / (1 + exp(-(mcS2-*s2_eff_par0) / *s2_eff_par1)));
		if (r3.Rndm() > 1. / (1 + exp(-(mcS2-*s2_eff_par0) / *s2_eff_par1))) continue;
		
		// s1 efficiency
		//printf("S1: %f \n", mcS1);
		//printf("eff: %f \n", 1. / (1 + exp(-(mcS1-*s1_eff_par0) / *s1_eff_par1)));
		if (r3.Rndm() > (1 - exp(-(mcS1-*s1_eff_par0) / *s1_eff_par1))) continue;
		
		
		aS1[i] = mcS1;
		aS2[i] = mcS2;
		
		
	}
}