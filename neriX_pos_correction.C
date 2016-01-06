#include "TProfile2D.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"

#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace std;

TFile fPositionCorrection = TFile("/Users/Matt/Desktop/Xenon/python_modules/src/pos_correction_run_15.root", "READ");
TProfile2D *hPosRec2DS1 = (TProfile2D *)fPositionCorrection.Get("pos_correction_S1");
TProfile2D *hPosRec2DS2 = (TProfile2D *)fPositionCorrection.Get("pos_correction_S2");

Double_t GetPosCorrectionS1(int run, float R, float Z)
{

	if (run == 15)
	{
		
		int bin = hPosRec2DS1->FindBin(pow(R, 2.), Z);
		double binContent = hPosRec2DS1->GetBinContent(bin);
		
		return binContent;
		
	}

	else
		return 1.0;
}


Double_t GetPosCorrectionS2(int run, float R, float Z)
{

	if (run == 15)
	{
		
		int bin = hPosRec2DS2->FindBin(pow(R, 2.), Z);
		double binContent = hPosRec2DS2->GetBinContent(bin);
		
		
		return binContent;
		
	}

	else
		return 1.0;
}



