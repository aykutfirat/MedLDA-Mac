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

#include "Params.h"
#include <string>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

Params::Params(void)
{
	train_filename = new char[512];
	test_filename = new char[512];
	PHI_DUALOPT = 1;
}

Params::~Params(void)
{
	delete[] train_filename;
	delete[] test_filename;
}

void Params::read_settings(char* filename)
{
	FILE* fileptr;
	char alpha_action[100];
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "var max iter %d\n", &VAR_MAX_ITER);
	fscanf(fileptr, "var convergence %f\n", &VAR_CONVERGED);
	fscanf(fileptr, "em max iter %d\n", &EM_MAX_ITER);
	fscanf(fileptr, "em convergence %f\n", &EM_CONVERGED);
	fscanf(fileptr, "model C %f\n", &INITIAL_C);
	fscanf(fileptr, "init alpha %f\n", &INITIAL_ALPHA);
	fscanf(fileptr, "svm_alg_type %d\n", &SVM_ALGTYPE);
	fscanf(fileptr, "alpha %d\n", &ESTIMATE_ALPHA);
	fscanf(fileptr, "phi-dual-opt %d\n", &PHI_DUALOPT);
	fscanf(fileptr, "inner-cv %s\n", alpha_action);
	if ( strcmp(alpha_action, "true") == 0 ) INNER_CV = true;
	else INNER_CV = false;

	fscanf(fileptr, "inner_foldnum %d\n", &INNER_FOLDNUM);
	fscanf(fileptr, "cv_paramnum %d\n", &CV_PARAMNUM);
	vec_cvparam = new double[CV_PARAMNUM];
	for ( int i=0; i<CV_PARAMNUM; i++ ) {
		float tmp;
		fscanf(fileptr, "%f\n", &tmp);
		vec_cvparam[i] = tmp;
	}

	fscanf(fileptr, "train_file: %s\n", train_filename);
	fscanf(fileptr, "test_file: %s\n", test_filename);

	fclose(fileptr);
}
