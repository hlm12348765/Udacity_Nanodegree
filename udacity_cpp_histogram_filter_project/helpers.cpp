#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream> 
//#include "debugging_helpers.cpp"

using namespace std;

vector< vector<float> > normalize(vector< vector <float> > grid) {
	
	float total = 0.0;
	int height = grid.size();
	int width = grid[0].size();
	vector< vector<float> > newGrid(height, vector<float>(width));

	for (int i  = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			total = total + grid[i][j]; 
		}
	}

	for (int k = 0; k < height; ++k) {
		for (int l = 0; l < width; ++l) {
			newGrid[k][l] = grid[k][l] / total;
		}
	}

	return newGrid;
}

vector < vector <float> > blur(vector < vector < float> > grid, float blurring) {

	vector < vector <float> > newGrid;
	int height = grid.size();
	int width = grid[0].size();
	float center_prob = 1 - blurring;
	float adjacent_prob = blurring / 6.0;
	float corner_prob = blurring / 12.0;
	vector < vector <float> > window (height, vector<float>(width));
	
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if ((i == 1) && (j == 1)) {
				window[i][j] = center_prob;
			}
			else if ((i == 1) && (j != 1)) {
				window[i][j] = adjacent_prob;
			}
			else if ((j == 1) && (i != 1)) {
				window[i][j] = adjacent_prob;
			}
			else {
				window[i][j] = corner_prob;
			}
		}
	}

	newGrid.assign(height, vector<float>(width, 0.0));
	int new_i, new_j;
	float mult;
	float grid_val;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			grid_val = grid[i][j];
			for (int dx = -1; dx < 2; ++dx) {
				for (int dy = -1; dy < 2; ++dy) {
					mult = window[dx+1][dy+1];
					if ((i + dy) >= 0) {
						new_i = (i + dy) % height;
					}
					else {
						new_i = (i + dy) % height + height;
					}
					if ((j + dx) >= 0) {
						new_j = (j + dx) % width;
					} 
					else {
						new_j = (j + dx) % width + width;
					}
					newGrid[new_i][new_j] = newGrid[new_i][new_j] + mult * grid_val;
				}
			}
		}
	}

	return normalize(newGrid);
}

bool close_enough(vector < vector <float> > g1, vector < vector <float> > g2) {
	int i, j;
	float v1, v2;
	if (g1.size() != g2.size()) {
		return false;
	}

	if (g1[0].size() != g2[0].size()) {
		return false;
	}
	for (i=0; i<g1.size(); i++) {
		for (j=0; j<g1[0].size(); j++) {
			v1 = g1[i][j];
			v2 = g2[i][j];
			if (abs(v2-v1) > 0.0001 ) {
				return false;
			}
		}
	}
	return true;
}

bool close_enough(float v1, float v2) { 
	if (abs(v2-v1) > 0.0001 ) {
		return false;
	} 
	return true;
}

vector <char> read_line(string s) {
	vector <char> row;

	size_t pos = 0;
	string token;
	string delimiter = " ";
	char cell;

	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		s.erase(0, pos + delimiter.length());

		cell = token.at(0);
		row.push_back(cell);
	}

	return row;
}

vector < vector <char> > read_map(string file_name) {
	ifstream infile(file_name);
	vector < vector <char> > map;
	if (infile.is_open()) {

		char color;
		vector <char> row;
		
		string line;

		while (std::getline(infile, line)) {
			row = read_line(line);
			map.push_back(row);
		}
	}
	return map;
}

vector < vector <float> > zeros(int height, int width) {
	int i, j;
	vector < vector <float> > newGrid;
	vector <float> newRow;

	for (i=0; i<height; i++) {
		newRow.clear();
		for (j=0; j<width; j++) {
			newRow.push_back(0.0);
		}
		newGrid.push_back(newRow);
	}
	return newGrid;
}

/*int main() {
	vector < vector < char > > map = read_map("maps/m1.txt");
 	show_grid(map);
 	return 0;
 }
*/