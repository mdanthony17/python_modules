#include "TMath.h"

using namespace std;

Double_t neriX_S1_efficiency(double g1, double S1)
{
	return TMath::BinomialI(g1, TMath::Nint(S1/g1), 1);
}


