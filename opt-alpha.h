#ifndef LDA_ALPHA_H
#define LDA_ALPHA_H

#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "utils.h"

#define NEWTON_THRESH 1e-5
#define MAX_ALPHA_ITER 1000

// for estimating alpha that is shared by all topics
double alhood(double a, double ss, int D, int K);
double d_alhood(double a, double ss, int D, int K);
double d2_alhood(double a, int D, int K);
double opt_alpha(double ss, int D, int K);

// for estimating alphas that are different for different topics
double alhood(double a, double alpha_sum, double ss, int D, int K);
double d_alhood(double a, double alpha_sum, double ss, int D, int K);
double d2_alhood(double a, double alpha_sum, int D, int K);
double opt_alpha(double ss, double &alpha_sum, int D, int K);

#endif
