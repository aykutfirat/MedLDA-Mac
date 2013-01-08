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

#include "utils.h"

/*
* given log(a) and log(b), return log(a + b)
*/
double log_sum(double log_a, double log_b)
{
	double v;

	if (log_a < log_b) {
		v = log_b+log(1 + exp(log_a-log_b));
	} else {
		v = log_a+log(1 + exp(log_b-log_a));
	}
	return(v);
}

/**
* Proc to calculate the value of the trigamma, the second
* derivative of the loggamma function. Accepts positive matrices.
* From Abromowitz and Stegun.  Uses formulas 6.4.11 and 6.4.12 with
* recurrence formula 6.4.6.  Each requires workspace at least 5
* times the size of X.
*
**/
double trigamma(double x)
{
	double p;
	int i;

	x = x+6;
	p = 1/(x*x);
	p = (((((0.075757575757576*p-0.033333333333333)*p+0.0238095238095238)
		*p-0.033333333333333)*p+0.166666666666667)*p+1)/x+0.5*p;
	for (i=0; i<6 ;i++) {
		x = x - 1;
		p = 1/(x*x)+p;
	}
	return(p);
}


/*
* taylor approximation of first derivative of the log gamma function
*/
double digamma(double x)
{
	double p;
	x = x+6;
	p = 1/(x*x);
	p = (((0.004166666666667*p-0.003968253986254)*p+
		0.008333333333333)*p-0.083333333333333)*p;
	p = p + log(x) - 0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	return p;
}


double log_gamma(double x)
{
	double z=1/(x*x);

	x=x+6;
	z=(((-0.000595238095238*z+0.000793650793651)
		*z-0.002777777777778)*z+0.083333333333333)/x;
	z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1)-
		log(x-2)-log(x-3)-log(x-4)-log(x-5)-log(x-6);
	return z;
}


double lgamma(double x)
{
	double x0,x2,xp,gl,gl0;
	int n,k;
	static double a[] = {
		8.333333333333333e-02,
		-2.777777777777778e-03,
		7.936507936507937e-04,
		-5.952380952380952e-04,
		8.417508417508418e-04,
		-1.917526917526918e-03,
		6.410256410256410e-03,
		-2.955065359477124e-02,
		1.796443723688307e-01,
		-1.39243221690590
	};

	x0 = x;
	if (x <= 0.0) return 1e308;
	else if ((x == 1.0) || (x == 2.0)) return 0.0;
	else if (x <= 7.0) {
		n = (int)(7-x);
		x0 = x+n;
	}
	x2 = 1.0/(x0*x0);
	xp = 2.0*M_PI;
	gl0 = a[9];
	for (k=8;k>=0;k--) {
		gl0 = gl0*x2 + a[k];
	}
	gl = gl0/x0+0.5*log(xp)+(x0-0.5)*log(x0)-x0;
	if (x <= 7.0) {
		for (k=1;k<=n;k++) {
			gl -= log(x0-1.0);
			x0 -= 1.0;
		}
	}
	return gl;
}

/*
 * make directory
 */
void make_directory(char* name)
{
    mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
}


/*
* argmax
*/
int argmax(double* x, int n)
{
	int i;
	double max = x[0];
	int argmax = 0;
	for (i = 1; i < n; i++)
	{
		if (x[i] > max)
		{
			max = x[i];
			argmax = i;
		}
	}
	return(argmax);
}

double dotprod(double *a, double *b, const int&n)
{
	double res = 0;
	for ( int i=0; i<n; i++ ) {
		res += a[i] * b[i];
	}
	return res;
}

void printmatrix(double **A, double n)
{
	for ( int i=0; i<n; i++ ) {
		for ( int j=0; j<n; j++ ) {
			printf("\t%f", A[i][j]);
		}
		printf("\n");
	}
}