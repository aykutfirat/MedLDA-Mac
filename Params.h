#pragma once

class Params
{
public:
	Params(void);
public:
	~Params(void);

	void read_settings(char* filename);

public:
	int LAG;

	float EM_CONVERGED;
	int EM_MAX_ITER;
	int ESTIMATE_ALPHA;
	float INITIAL_ALPHA;
	float INITIAL_C;
	int NTOPICS;
	int NLABELS;
	int NFOLDS;
	int FOLDIX;
	bool INNER_CV;
	float DELTA_ELL;
	int PHI_DUALOPT;

	int VAR_MAX_ITER;
	float VAR_CONVERGED;
	double *vec_cvparam;
	int INNER_FOLDNUM;
	int CV_PARAMNUM;

	int SVM_ALGTYPE;       // the algorithm type for SVM

	char *train_filename;   // the file names of training & testing data sets
	char *test_filename;
};
