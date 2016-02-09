#include <vector>
#include <functional>
#include <numeric>

// sometimes numpy dot product is not thread safe
// use this in its place if that is the case

vector <double> safe_dot(double *aUnraveledMatrix, double *aBeforeValues, int numRowsMatrix, int numColumns)
{
	vector<double> aAfterValues(numRowsMatrix, 0.);
	for (int rowNumber=0; rowNumber < numRowsMatrix; rowNumber++)
	{
		aAfterValues[rowNumber] = std::inner_product(aUnraveledMatrix + (rowNumber+0)*numColumns, aUnraveledMatrix + (rowNumber+1)*numColumns, aBeforeValues, 0.);
	}
	return aAfterValues;
}