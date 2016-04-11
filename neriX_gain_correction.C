#include "TF1.h"
#include "TGraph.h"
#include "TFile.h"

#include <vector>
#include <stdio.h>
#include <iostream>

using namespace std;

Double_t GetGainCorrectionBottomPMT(int run, double unixtime)
{

	if (run == 10)
	{
		if ((unixtime-1.41069584e+09) >= 74160 && (unixtime-1.41069584e+09) < 1906360)
			return (((unixtime-1.41069584e+09)*(-0.0496189113) + 721397.248)/584000.);
		if ((unixtime-1.41069584e+09) >= 1906360 && (unixtime-1.41069584e+09) < 4500160)
			return (((unixtime-1.41069584e+09)*(-0.00761140539) + 647299.659)/584000.);
		else
			return 1.0;
	}
	
	else if (run == 11)
	{
		if ((unixtime-1.41641345e+09) >= 1042584 && (unixtime-1.41641345e+09) < 1405154)
			return (((unixtime-1.41641345e+09)*(-0.159038062) + 716227.799)/579000.);
		if ((unixtime-1.41641345e+09) >= 1405154 && (unixtime-1.41641345e+09) < 2612294)
			return (((unixtime-1.41641345e+09)*(-0.0282061688) + 510797.205)/579000.);
		if ((unixtime-1.41641345e+09) >= 2617454 && (unixtime-1.41641345e+09) < 4082205)
			return (((unixtime-1.41641345e+09)*(-0.115794313) + 1194109.02)/847000.);
		if ((unixtime-1.41641345e+09) >= 4082205 && (unixtime-1.41641345e+09) < 4860402)
			return (((unixtime-1.41641345e+09)*(-0.0309546184) + 855894.658)/847000.);
		if ((unixtime-1.41641345e+09) >= 4860402 && (unixtime-1.41641345e+09) < 6652057)
			return (((unixtime-1.41641345e+09)*(-0.00692265175) + 741803.435)/847000.);
		else
			return 1.0;
	}
	else
		return 1.0;
}

Double_t GetGainCorrectionErrorBottomPMT(int run, double unixtime)
{
	if (run == 10)
	{
		if ((unixtime-1.41069584e+09) >= 74160 && (unixtime-1.41069584e+09) < 1906360)
			return (sqrt(pow((unixtime-1.41069584e+09)*(0.0126776427),2) + pow(15569.6786,2) + 2.*(unixtime-1.41069584e+09)*(-185.067542))/584000.);
		if ((unixtime-1.41069584e+09) >= 1906360 && (unixtime-1.41069584e+09) < 4500160)
			return (sqrt(pow((unixtime-1.41069584e+09)*(0.00646620325),2) + pow(21542.8309,2) + 2.*(unixtime-1.41069584e+09)*(-135.723929))/584000.);
		else
			return 1.0;
	}
	else if (run == 11)
	{
		if ((unixtime-1.41641345e+09) >= 1042584 && (unixtime-1.41641345e+09) < 1405154)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0850958814),2) + pow(102758.583,2) + 2.*(unixtime-1.41641345e+09)*(-8696.7309))/579000.);
		if ((unixtime-1.41641345e+09) >= 1405154 && (unixtime-1.41641345e+09) < 2612294)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0306176609),2) + pow(64921.7002,2) + 2.*(unixtime-1.41641345e+09)*(-1967.76783))/579000.);
		if ((unixtime-1.41641345e+09) >= 2617454 && (unixtime-1.41641345e+09) < 4082205)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0376555224),2) + pow(132418.41,2) + 2.*(unixtime-1.41641345e+09)*(-4944.81036))/847000.);
		if ((unixtime-1.41641345e+09) >= 4082205 && (unixtime-1.41641345e+09) < 4860402)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0532606209),2) + pow(240915.163,2) + 2.*(unixtime-1.41641345e+09)*(-12804.588))/847000.);
		if ((unixtime-1.41641345e+09) >= 4860402 && (unixtime-1.41641345e+09) < 6652057)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0218790998),2) + pow(125776.959,2) + 2.*(unixtime-1.41641345e+09)*(-2731.76023))/847000.);
		else
			return 1.0;
	}
	else
		return 1.0;
}


