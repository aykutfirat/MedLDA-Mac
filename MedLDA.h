#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "opt-alpha.h"
#include "utils.h"
#include "cokus.h"
#include "Params.h"
#include "Corpus.h"
#include "svm_multiclass/svm_struct_api.h"
#include "svm_multiclass/svm_struct_learn.h"
#include "svm_multiclass/svm_struct_common.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 1

#define NEWTON_THRESH 1e-5
#define MAX_ALPHA_ITER 1000

typedef struct
{
    double **class_word;
    double *class_total;
    double *alpha_suffstats;
    int num_docs;

	// for supervised LDA
	double **exp;	  // E[zz^\top]
	int *y;			  // the vector of response value
	char dir[512];
} SuffStats;


class MedLDA
{
public:
	MedLDA(void);
public:
	~MedLDA(void);

public:

	double doc_e_step(Document* doc, double* gamma, double** phi, SuffStats* ss, Params *param);

	bool mle(SuffStats* ss, Params *param, bool bInit = true);


	void save_gamma(char* filename, double** gamma, int num_docs, int num_topics);
	double save_prediction(char *filename, Corpus *corpus);

	int run_em(char* start, char* directory, Corpus* corpus, Params* param);

	double infer(char* model_dir, Corpus* corpus, Params *param);

	double inference(Document*, const int&, double*, double **, Params* param);
	double inference_pred(Document*, double*, double**, Params* param);
	double compute_mrgterm(Document *doc, const int &d, const int &n, const int &k, Params* param);
	double compute_lhood(Document*, double**, double*);

	void partitionData(Corpus *corpus, double** gamma, int ntopic, int nLabels);
	void outputData(char *filename, Corpus *corpus, double** gamma, int ntopic, int nLabels);
	void outputData2(char *filename, Corpus *corpus, double **exp, int ntopic, int nLabels);
	double innerCV(char *modelDir, Corpus *c);
	void predict(Document *doc, double **phi);
	void loss_aug_predict(Document *doc, double *zbar_mean);
	double loss(const int &y, const int &gnd);

	void free_model();
	void save_model(char*);
	void new_model(int, int, int, int, double);
	SuffStats* new_suffstats();
	void corpus_init_ss(SuffStats* ss, Corpus* c);
	void random_init_ss(SuffStats* ss, Corpus* c);
	void zero_init_ss(SuffStats* ss);
	void load_model(char* model_root);

	void set_init_param(STRUCT_LEARN_PARM *struct_parm,
					LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,
					int *alg_type);
	void svmStructSolver(SuffStats* ss, Params *param, double *res);
	void outputLowDimData(char *filename, SuffStats *ss);

	void write_word_assignment(FILE* f, Document* doc, double** phi);

public:
	int m_nK;
	int m_nLabelNum;
	int m_nNumTerms;
	double **m_dLogProbW;
	double m_dDeltaEll;   // adjustable 0/ell loss function

private:
	double *m_alpha;
	double *m_dMu;
	double *m_dEta;
	double m_dC;
	double m_dB;
	int m_nDim;
	double m_dsvm_primalobj;
};
