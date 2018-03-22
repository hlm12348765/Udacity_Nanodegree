#include "headers/zeros.h"

using namespace std;

vector < vector <float> > zeros(int height, int width) {
  
	// OPTIMIZATION: Reserve space in memory for vectors
	vector < vector <float> > newGrid;
	vector <float> newRow;
	newGrid.reserve(height);
	newRow.assign(width, 0);

  	// OPTIMIZATION: nested for loop not needed
    // because every row in the matrix is exactly the same
	/*for (int i=0; i<height; i++) {
		newRow.clear();
		for (int j=0; j<width; j++) {
			newRow.push_back(0.0);
		}
		newGrid.push_back(newRow);
	}*/
	for (int i = 0; i < height; ++i) {
		newGrid.push_back(newRow);
	}

	return newGrid;
}