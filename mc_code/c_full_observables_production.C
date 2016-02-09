#include <vector>
#include <TH1.h>
#include <TRandom3.h>
#include <stdio.h>
#include <math.h>

void full_matching_loop(int *seed, int *numTrials, float *aS1, float *aS2, float *aEnergy, float *photonYield, float *chargeYield, float *excitonToIonRatio, float *g1Value, float *extractionEfficiency, float *gasGainValue, float *gasGainWidth, float *speRes, float *intrinsicResS1, float *intrinsicResS2)
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
	
	float probRecombination = ( (*excitonToIonRatio+1) * *photonYield )/(*photonYield+*chargeYield) - *excitonToIonRatio;
	//printf("%f \\n", probRecombination);

	for (int i = 0; i < *numTrials; i++)
	{
		// ------------------------------------------------
		//  Draw random energy from distribution
		// ------------------------------------------------
		
		mcEnergy = aEnergy[i];
		//printf("%f \\n", mcEnergy);


		// ------------------------------------------------
		//  Find number of quanta
		// ------------------------------------------------
		
		
		mcQuanta = r3.Poisson(mcEnergy*(*photonYield + *chargeYield));
		//printf("%d \\n", numQuanta);
		
		
		// ------------------------------------------------
		//  Convert to excitons and ions
		// ------------------------------------------------
		
		
		probExcitonSuccess = 1. - 1./(1. + *excitonToIonRatio);
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
		mcExtractedElectrons = r3.Binomial(mcElectrons, *extractionEfficiency);
		mcS2 = r3.Gaus(mcExtractedElectrons * *gasGainValue, *gasGainWidth*pow(mcExtractedElectrons, 0.5));
		
		if (mcS1 < 0) continue;
		if (mcS2 < 0) continue;
		
		//printf("%f \\n", mcS1);
		//printf("%f \\n", mcS2);
		
		
		// ------------------------------------------------
		//  Smear S1 and S2
		// ------------------------------------------------
		
		if (*speRes <= 0 || *intrinsicResS1 <= 0 || *intrinsicResS2 <= 0) continue;
		
		mcS1 = r3.Gaus(mcS1, *speRes*pow(mcS1, 0.5));
		if (mcS1 < 0) continue;
		mcS1 = r3.Gaus(mcS1, *intrinsicResS1*pow(mcS1, 0.5));
		if (mcS1 < 0) continue;
		
		mcS2 = r3.Gaus(mcS2, *speRes*pow(mcS2, 0.5));
		if (mcS2 < 0) continue;
		mcS2 = r3.Gaus(mcS2, *intrinsicResS2*pow(mcS2, 0.5));
		if (mcS2 < 0) continue;
		
		aS1[i] = mcS1;
		aS2[i] = mcS2;
		
		
	}
}