#include "headers/blur.h"
#include "headers/zeros.h"

using namespace std;

// OPTIMIZATION: Pass large variable by reference
vector < vector <float> > blur(vector < vector < float> > grid, float blurring) {

	// OPTIMIZATION: window, DX and  DY variables have the 
    // same value each time the function is run.
  	// It's very inefficient to recalculate the vectors
    // every time the function runs. 
    // 
    // The const and/or static operator could be useful.
  	// Define and declare window, DX, and DY using the
    // bracket syntax: vector<int> foo = {1, 2, 3, 4} 
    // instead of calculating these vectors with for loops 
    // and push back
	vector < vector <float> > newGrid;
	vector <float> row;
	vector <float> newRow;

	int height;
	int width;
	float center, corner, adjacent;

	height = grid.size();
	width = grid[0].size();

	center = 1.0 - blurring;
	corner = blurring / 12.0;
	adjacent = blurring / 6.0;

	static vector < vector <float> > window = {
		{corner, adjacent, corner},
        {adjacent, center, adjacent},
        {corner, adjacent, corner}
	};

	static vector <int> DX = {-1, 0, 1};
	static vector <int> DY = {-1, 0, 1};

	int ii;
	int jj;
	int new_i;
	int new_j;

	// OPTIMIZATION: Use your improved zeros function
	newGrid = zeros(height, width);

	for (int i=0; i< height; i++ ) {
		for (int j=0; j<width; j++ ) {
			for (int ii=0; ii<3; ii++) {
				for (int jj=0; jj<3; jj++) {
					new_i = (i + DY[ii] + height) % height;
					new_j = (j + DX[jj] + width) % width;
					newGrid[new_i][new_j] += grid[i][j] * window[ii][jj];
				}
			}
		}
	}

	return newGrid;
}
