// (C) Copyright 2009, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu)

// This file is part of MedLDA.

// MedLDA is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// MedLDA is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "opt-alpha.h"

/*
 * case 1: different topics share the same prior
 *   Newton's method with the objective function and its derivatives
 */
double alhood(double a, double ss, int D, int K)
{ return(D * (lgamma(K * a) - K * lgamma(a)) + (a - 1) * ss); }

double d_alhood(double a, double ss, int D, int K)
{ return(D * (K * digamma(K * a) - K * digamma(a)) + ss); }

double d2_alhood(double a, int D, int K)
{ return(D * (K * K * trigamma(K * a) - K * trigamma(a))); }

double opt_alpha(double ss, int D, int K)
{
    double a, log_a, init_a = 100;
    double f, df, d2f;
    int iter = 0;

    log_a = log(init_a);
    do
    {
        iter++;
        a = exp(log_a);
        if (/*isnan(a)*/a>1e300 || a<1e-300)
        {
            init_a = init_a * 10;
            printf("warning : alpha is nan; new init = %5.5f\n", init_a);
            a = init_a;
            log_a = log(a);
        }
        f = alhood(a, ss, D, K);
        df = d_alhood(a, ss, D, K);
        d2f = d2_alhood(a, D, K);
        log_a = log_a - df/(d2f * a + df);
        printf("alpha maximization : %5.5f   %5.5f\n", f, df);
    }
    while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
    return(exp(log_a));
}


/*
 * case 2: different topics have different priors
 *   Newton's method with the objective function and its derivatives
 */
double alhood(double a, double alpha_sum, double ss, int D, int K)
{ return(D * (lgamma(alpha_sum) - lgamma(a)) + (a - 1) * ss); }

double d_alhood(double a, double alpha_sum, double ss, int D, int K)
{ return(D * (digamma(alpha_sum) - digamma(a)) + ss); }

double d2_alhood(double a, double alpha_sum, int D, int K)
{ return(D * (trigamma(alpha_sum) - trigamma(a))); }

double opt_alpha(double ss, double &alpha_sum, int D, int K)
{
    double a, log_a, init_a = 100, old_a = 0;
    double f, df, d2f;
    int iter = 0;

    log_a = log(init_a);
    do
    {
        iter++;
        a = exp(log_a);
		alpha_sum += a - old_a;
		old_a = a;
        if (/*isnan(a)*/a>1e300 || a<1e-300)
        {
            init_a = init_a * 10;
            printf("warning : alpha is nan; new init = %5.5f\n", init_a);
            a = init_a;
            log_a = log(a);
        }
        f = alhood(a, alpha_sum, ss, D, K);
        df = d_alhood(a, alpha_sum, ss, D, K);
        d2f = d2_alhood(a, alpha_sum, D, K);
        log_a = log_a - df/(d2f * a + df);
        //printf("alpha maximization : %5.5f   %5.5f\n", f, df);
    }
    while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
    return(exp(log_a));
}
