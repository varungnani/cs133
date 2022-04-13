// Header inclusions, if any...
#include "lib/gemm.h"
#include <cmath>
#include <cstring>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>


// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
  omp_set_num_threads(32);
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  //transpose matrix

  float[kK][kJ] b_trans;
  for (int i = 0; i < kK; i++) {
      for (int j = 0; j < kJ; j++) {
          b_trans[i][j] = b[j][i];
      }
  }


  int i;
  int j;
  int k;
  
  #pragma omp parallel for shared(a,b_trans,c) private(i,j,k) schedule(guided)
  for (i = 0; i < kI; ++i) {
    for (j = 0; j < kJ; ++j) {
      for (k = 0; k < kK; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  // Your code goes here...
}
