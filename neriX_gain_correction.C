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
			return (((unixtime-1.41069584e+09)*(-0.0595022361) + 730824.682)/584000.);
		if ((unixtime-1.41069584e+09) >= 1906360 && (unixtime-1.41069584e+09) < 4500160)
			return (((unixtime-1.41069584e+09)*(-0.00830179934) + 635470.922)/584000.);
		else
			return 1.0;
	}
	
	else if (run == 11)
	{
	if ((unixtime-1.41641345e+09) >= 1042584 && (unixtime-1.41641345e+09) < 1405154)
		return (((unixtime-1.41641345e+09)*(-0.163279122) + 737115.668)/579000.);
	if ((unixtime-1.41641345e+09) >= 1405154 && (unixtime-1.41641345e+09) < 2612294)
		return (((unixtime-1.41641345e+09)*(-0.0290143145) + 526036.5)/579000.);
	if ((unixtime-1.41641345e+09) >= 2617454 && (unixtime-1.41641345e+09) < 4082205)
		return (((unixtime-1.41641345e+09)*(-0.117644109) + 1213184.16)/847000.);
	if ((unixtime-1.41641345e+09) >= 4082205 && (unixtime-1.41641345e+09) < 4860402)
		return (((unixtime-1.41641345e+09)*(-0.0314297512) + 869495.328)/847000.);
	if ((unixtime-1.41641345e+09) >= 4860402 && (unixtime-1.41641345e+09) < 6652057)
		return (((unixtime-1.41641345e+09)*(-0.00703372361) + 753657.931)/847000.);
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
			return (sqrt(pow((unixtime-1.41069584e+09)*(0.00958679193),2) + pow(12223.0126,2) + 2.*(unixtime-1.41069584e+09)*(-109.965924))/584000.);
		if ((unixtime-1.41069584e+09) >= 1906360 && (unixtime-1.41069584e+09) < 4500160)
			return (sqrt(pow((unixtime-1.41069584e+09)*(0.00446434401),2) + pow(15109.6245,2) + 2.*(unixtime-1.41069584e+09)*(-65.5991024))/584000.);
		else
			return 1.0;
	}
	else if (run == 11)
	{
		if ((unixtime-1.41641345e+09) >= 1042584 && (unixtime-1.41641345e+09) < 1405154)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.13618288),2) + pow(164621.918,2) + 2.*(unixtime-1.41641345e+09)*(-22297.5046))/579000.);
		if ((unixtime-1.41641345e+09) >= 1405154 && (unixtime-1.41641345e+09) < 2612294)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0494684194),2) + pow(104777.428,2) + 2.*(unixtime-1.41641345e+09)*(-5131.99029))/579000.);
		if ((unixtime-1.41641345e+09) >= 2617454 && (unixtime-1.41641345e+09) < 4082205)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0375003386),2) + pow(131873.504,2) + 2.*(unixtime-1.41641345e+09)*(-4904.17708))/847000.);
		if ((unixtime-1.41641345e+09) >= 4082205 && (unixtime-1.41641345e+09) < 4860402)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0530365344),2) + pow(239896.618,2) + 2.*(unixtime-1.41641345e+09)*(-12696.7912))/847000.);
		if ((unixtime-1.41641345e+09) >= 4860402 && (unixtime-1.41641345e+09) < 6652057)
			return (sqrt(pow((unixtime-1.41641345e+09)*(0.0217889325),2) + pow(125249.584,2) + 2.*(unixtime-1.41641345e+09)*(-2709.08864))/847000.);
		else
			return 1.0;
	}
	else
		return 1.0;
}


