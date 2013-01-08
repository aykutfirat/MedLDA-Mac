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

#include <vector>
#include <string>
#include "svmlight/svm_common.h"
#include "MedLDA.h"
#include "Corpus.h"
using namespace std;


double innerCV(char *modelDir, Corpus *c, Params *param);

/*
* main
*/
int main(int argc, char* argv[])
{
	seedMT( time(NULL) );
	// seedMT(4357U);

	if (argc > 1)
	{
		Corpus* c = new Corpus();
		Params param;
		param.INNER_CV = true;
		if ( strcmp(argv[1], "estinf") == 0 ) {
			param.read_settings("settings.txt");
			param.NTOPICS = atoi(argv[2]);
			param.NLABELS = atoi(argv[3]);
			param.NFOLDS = atoi(argv[4]);
			param.INITIAL_C = atof(argv[5]);
			param.DELTA_ELL = atof(argv[6]);

			printf("K: %d, C: %.3f, Alpha: %d, svm: %d\n", param.NTOPICS, 
				param.INITIAL_C, param.ESTIMATE_ALPHA, param.SVM_ALGTYPE);

			c->read_data(param.train_filename, param.NLABELS);
			char dir[512];
			sprintf(dir, "20ng%d_c%d_f%d", param.NTOPICS, (int)param.INITIAL_C, param.NFOLDS);
			make_directory(dir);

			if ( param.INNER_CV ) {
				c->shuffle();

				char modelDir[512];
				sprintf(modelDir, "%s/innercv", dir);
				make_directory(modelDir);

				param.INITIAL_C = innerCV(modelDir, c, &param);
				printf("\n\nBest C: %f\n", param.INITIAL_C);
			}
			MedLDA model;
			model.run_em(argv[7], dir, c, &param);

			// testing.
			Corpus *tstC = new Corpus();
			tstC->read_data(param.test_filename, param.NLABELS);
			MedLDA evlModel;
			double dAcc = evlModel.infer(dir, tstC, &param);
			printf("Accuracy: %.3f\n", dAcc);
			delete tstC;
		}
		if ( strcmp(argv[1], "est") == 0 ) {
			param.read_settings("settings.txt");
			param.NTOPICS = atoi(argv[2]);
			param.NLABELS = atoi(argv[3]);
			param.NFOLDS = atoi(argv[4]);
			param.INITIAL_C = atof(argv[5]);
			param.DELTA_ELL = atof(argv[6]);

			c->read_data(param.train_filename, param.NLABELS);
			char dir[512];
			sprintf(dir, "%s%d_c%d_f%d", argv[7], param.NTOPICS, param.INITIAL_C, param.NFOLDS);
			make_directory(dir);

			if ( param.INNER_CV ) {
				c->shuffle();

				char modelDir[512];
				sprintf(modelDir, "%s/innercv", dir);
				make_directory(modelDir);

				param.INITIAL_C = innerCV(modelDir, c, &param);
				printf("\n\nBest C: %f\n", param.INITIAL_C);
			}
			MedLDA model;
			model.run_em(argv[8], dir, c, &param);
		}
		if (strcmp(argv[1], "inf")==0)
		{
			param.read_settings("settings.txt");
			param.NLABELS = atoi(argv[2]);
			c->read_data(param.test_filename, param.NLABELS);
			MedLDA model;
			double dAcc = model.infer(argv[3], c, &param);
			printf("Accuracy: %.3f\n", dAcc);
		}

		delete c;
	} else {
		printf("usage : MEDsLDAc estinf [k] [labels] [fold] [initial C] [l] [random/seeded/*]\n");
		printf("        MEDsLDAc est [k] [labels] [fold] [initial C] [l] [dir root] [random/seeded/*]\n");
		printf("        MEDsLDAc inf [labels] [model]\n");
	}
	return(0);
}

double innerCV(char *modelDir, Corpus *c, Params *param)
{
	int nMaxUnit = c->num_docs / 5 + 5;

	double dBestAccuracy = 0;
	double dBestC = 0;
	for ( int k=0; k<param->CV_PARAMNUM; k++ )
	{
		printf("\n\n$$$ Learning with C: %.4f $$$ \n\n", param->vec_cvparam[k]);

		param->INITIAL_C = param->vec_cvparam[k];
		double dAvgAccuracy = 0;
		for ( int i=1; i<=param->INNER_FOLDNUM; i++ )
		{
			// get training & test data
			Corpus *trDoc = c->get_traindata(param->INNER_FOLDNUM, i);
			Corpus *tstDoc = c->get_testdata(param->INNER_FOLDNUM, i);

			MedLDA model;
			model.run_em("random", modelDir, trDoc, param);

			// predict on test corpus
			MedLDA evlModel;
			double dAcc = evlModel.infer(modelDir, tstDoc, param);
			dAvgAccuracy += dAcc / param->INNER_FOLDNUM;

			delete trDoc;
			delete tstDoc;
			printf("\n\n");
		}
		printf("@@@ Avg Accuracy: %.4f\n\n", dAvgAccuracy);

		if ( dAvgAccuracy > dBestAccuracy )	{
			dBestC = param->vec_cvparam[k];
			dBestAccuracy = dAvgAccuracy;
		}
	}
	
	return dBestC;
}
