#include "helpers.cpp"
#include <stdlib.h>
#include "debugging_helpers.cpp"

using namespace std;

vector< vector <float> > initialize_beliefs(vector< vector <char> > grid) {

	vector< vector <float> > newGrid;
	int height = grid.size();
	int width = grid[0].size();
	float area = height * width;
	float belief_per_cell = 1.0 / area;

	for (int i = 0; i < height; ++i) {
		vector<float> row;
		for (int j = 0; j < width; ++j) {
			row.push_back(belief_per_cell);
		}
		newGrid.push_back(row);
	}
	
	return newGrid;
}

vector< vector <float> > sense(char color, 
	vector< vector <char> > grid, 
	vector< vector <float> > beliefs, 
	float p_hit,
	float p_miss) 
{
	int height = beliefs.size();
	int width = beliefs[0].size();
	vector< vector <float> > newGrid(height, vector<float>(width));
	for (int i =0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int hit = (color == grid[i][j]);
			newGrid[i][j] = beliefs[i][j] * (hit * p_hit + (1 - hit) * p_miss);
		}
	}

	return normalize(newGrid);
}

vector< vector <float> > move(int dy, int dx, 
	vector < vector <float> > beliefs,
	float blurring) 
{
	int height = beliefs.size();
	int width = beliefs[0].size();
	vector < vector <float> > newGrid(height, vector<float>(width));

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int new_i = (i + dy) % height;
			int new_j = (j + dy) % width;
			newGrid[new_i][new_j] = beliefs[i][j];
		}
	}

	return blur(newGrid, blurring);
}