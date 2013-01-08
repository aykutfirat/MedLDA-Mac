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

#include "MedLDA.h"
#include <vector>
#include <string>
#include "svmlight/svm_common.h"
#include "Params.h"
using namespace std;

MedLDA::MedLDA(void)
{
	m_dLogProbW = NULL;
	m_dMu       = NULL;
	m_dEta      = NULL;
	m_alpha     = NULL;
}

MedLDA::~MedLDA(void)
{
	free_model();
}

/*
* perform inference on a Document and update sufficient statistics
*/
double MedLDA::doc_e_step(Document* doc, double* gamma, double** phi,
				  SuffStats* ss, Params *param)
{
	// posterior inference
	double lhood = inference(doc, ss->num_docs, gamma, phi, param);

	// update sufficient statistics
	double gamma_sum = 0;
	for (int k=0; k<m_nK; k++) {
		gamma_sum += gamma[k];
		ss->alpha_suffstats[k] += digamma(gamma[k]);
	}
	for ( int k=0; k<m_nK; k++ ) {
		ss->alpha_suffstats[k] -= /*m_nK * */digamma(gamma_sum);
	}

	for (int k = 0; k < m_nK; k++) {
		double dVal = 0;
		for (int n = 0; n < doc->length; n++) {
			ss->class_word[k][doc->words[n]] += doc->counts[n]*phi[n][k];
			ss->class_total[k] += doc->counts[n]*phi[n][k];
			dVal += phi[n][k] * doc->counts[n] / doc->total;
		}
	
		// suff-stats for supervised LDA
		ss->exp[ss->num_docs][k] = dVal;
	}
	ss->num_docs = ss->num_docs + 1;

	return lhood;
}

/*
 * variational inference
 */
double MedLDA::inference(Document* doc, const int &docix, 
						 double* var_gamma, double** phi, Params *param)
{
    double converged = 1;
    double phisum = 0, lhood = 0;
    double lhood_old = 0;
	double *oldphi = (double*)malloc(sizeof(double)*m_nK);
    double *digamma_gam = (double*)malloc(sizeof(double)*m_nK);

    // compute posterior dirichlet
    for (int k = 0; k < m_nK; k++) {
        var_gamma[k] = m_alpha[k] + (doc->total/((double) m_nK));
        digamma_gam[k] = digamma(var_gamma[k]);
        for (int n = 0; n < doc->length; n++) {
            phi[n][k] = 1.0/m_nK;
		}
    }

	int var_iter = 0;
	//FILE *fileptr = fopen("PhiDist.txt", "w");
    while ((converged > param->VAR_CONVERGED) && ((var_iter < param->VAR_MAX_ITER) 
		|| (param->VAR_MAX_ITER == -1)))
    {
		var_iter ++;
		for (int n = 0; n < doc->length; n++) {
			phisum = 0; 

			if ( param->PHI_DUALOPT != 1 ) loss_aug_predict(doc, var_gamma); // loss-augmented prediction

			for (int k = 0; k < m_nK; k++) {
				oldphi[k] = phi[n][k];
				
				/* update the phi: add additional terms here for supervised LDA */
				double dVal = compute_mrgterm(doc, docix, n, k, param);
				phi[n][k] =	digamma_gam[k] + m_dLogProbW[k][doc->words[n]]  // the following two terms for sLDA
								+ dVal;

				//fprintf(fileptr, "%.5f:%.5f ", digamma_gam[k] + m_dLogProbW[k][doc->words[n]], dVal);
				if (k > 0) phisum = log_sum(phisum, phi[n][k]);
				else       phisum = phi[n][k]; // note, phi is in log space
			}
			//fprintf(fileptr, "\n");

			// update gamma and normalize phi
			for (int k = 0; k < m_nK; k++) {
				phi[n][k] = exp(phi[n][k] - phisum);
				var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);

				digamma_gam[k] = digamma(var_gamma[k]);
			}
		}

		lhood = compute_lhood(doc, phi, var_gamma);
		//assert(!isnan(lhood));
		converged = (lhood_old - lhood) / lhood_old;
		lhood_old = lhood;
    }
	//fclose(fileptr);

	free(oldphi);
    free(digamma_gam);

    return(lhood);
}

// find the loss-augmented prediction for one document.
void MedLDA::loss_aug_predict(Document *doc, double *zbar_mean)
{
	doc->lossAugLabel = -1;
	double dMaxScore = 0;
	for ( int y=0; y<m_nLabelNum; y++ )
	{
		double dScore = 0;
		for ( int k=0; k<m_nK; k++ ) {
			int etaIx = y * m_nK + k;
			dScore += zbar_mean[k] * m_dEta[etaIx];
		}
		dScore -= m_dB;
		dScore += loss(y, doc->gndlabel);

		if ( doc->lossAugLabel == -1 || dScore > dMaxScore ) {
			doc->lossAugLabel = y;
			dMaxScore = dScore;
		}
	}	
}

double MedLDA::compute_mrgterm(Document *doc, const int &d, const int &n, const int &k, Params *param)
{
	double dval = 0;
	int gndetaIx = doc->gndlabel * m_nK + k;
	if ( param->PHI_DUALOPT = 1 ) {  // use the dual parameters
		for ( int m=0; m<m_nLabelNum; m++ ) {
			int muIx = d * m_nLabelNum + m;
			int etaIx = m * m_nK + k;

			dval += m_dMu[muIx] * (m_dEta[gndetaIx] - m_dEta[etaIx]);
		}
	} else {                     // use the most-violated constraints
		int etaIx = doc->lossAugLabel * m_nK + k;
		dval = m_dC * (m_dEta[gndetaIx] - m_dEta[etaIx]);
	}
	dval = dval * doc->counts[n] / doc->total;
	
	return dval;
}

double MedLDA::loss(const int &y, const int &gnd)
{
	if ( y == gnd ) return 0;
	else return m_dDeltaEll;
}

/* 
* Given the model and w, compute the E[Z] for prediction
*/
double MedLDA::inference_pred(Document* doc, double* var_gamma, double** phi, Params *param)
{
    double converged = 1;
    double phisum = 0, lhood = 0;
    double lhood_old = 0;
	double *oldphi = (double*)malloc(sizeof(double)*m_nK);
    int k, n, var_iter;
    double *digamma_gam = (double*)malloc(sizeof(double)*m_nK);

    // compute posterior dirichlet
    for (k = 0; k < m_nK; k++) {
        var_gamma[k] = m_alpha[k] + (doc->total/((double) m_nK));
        digamma_gam[k] = digamma(var_gamma[k]);
        for (n = 0; n < doc->length; n++) {
            phi[n][k] = 1.0/m_nK;
		}
    }
    var_iter = 0;


    while ((converged > param->VAR_CONVERGED) && ((var_iter < param->VAR_MAX_ITER) 
		|| (param->VAR_MAX_ITER == -1)))
    {
		var_iter ++;
		for (n = 0; n < doc->length; n++)
		{
			phisum = 0; 
			for (k = 0; k < m_nK; k++) {
				oldphi[k] = phi[n][k];
				
				phi[n][k] =	digamma_gam[k] + m_dLogProbW[k][doc->words[n]];

				if (k > 0) phisum = log_sum(phisum, phi[n][k]);
				else       phisum = phi[n][k]; // note, phi is in log space
			}

			// update gamma and normalize phi
			for (k = 0; k < m_nK; k++) {
				phi[n][k] = exp(phi[n][k] - phisum);
				var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
				// !!! a lot of extra digamma's here because of how we're computing it
				// !!! but its more automatically updated too.
				digamma_gam[k] = digamma(var_gamma[k]);
			}
		}

		lhood = compute_lhood(doc, phi, var_gamma);
		converged = (lhood_old - lhood) / lhood_old;
		lhood_old = lhood;
    }

	free(oldphi);
    free(digamma_gam);

    return(lhood);
}

/*
 * compute lhood bound
 */
double MedLDA::compute_lhood(Document* doc, double** phi, double* var_gamma)
{
	double lhood = 0, digsum = 0, var_gamma_sum = 0, alpha_sum = 0;
	double *dig = (double*)malloc(sizeof(double)*m_nK);

	for (int k = 0; k < m_nK; k++) {
		dig[k] = digamma(var_gamma[k]);
		var_gamma_sum += var_gamma[k];
		alpha_sum += m_alpha[k];
	}
	digsum = digamma(var_gamma_sum);

	lhood = lgamma( alpha_sum ) - (lgamma(var_gamma_sum));
	for ( int k=0; k<m_nK; k++ ) {
		lhood -= lgamma( m_alpha[k] );
	}

	for (int k = 0; k < m_nK; k++) {
		lhood += (m_alpha[k] - 1)*(dig[k] - digsum) + lgamma(var_gamma[k])
			- (var_gamma[k] - 1)*(dig[k] - digsum);

		double dVal = 0;
		for (int n = 0; n < doc->length; n++) {
			if (phi[n][k] > 0) {
				lhood += doc->counts[n] * (phi[n][k]*((dig[k] - digsum) - log(phi[n][k])
					+ m_dLogProbW[k][doc->words[n]]));
			}
			dVal += phi[n][k] * doc->counts[n] / doc->total;
		}
	}

	free(dig);
	return(lhood);
}

/*
* writes the word assignments line for a Document to a file
*
*/
void MedLDA::write_word_assignment(FILE* f, Document* doc, double** phi)
{
	fprintf(f, "%03d", doc->length);
	for (int n = 0; n < doc->length; n++) {
		fprintf(f, " %04d:%02d", doc->words[n], argmax(phi[n], m_nK));
	}
	fprintf(f, "\n");
	fflush(f);
}


/*
* saves the gamma parameters of the current dataset
*/
void MedLDA::save_gamma(char* filename, double** gamma, int num_docs, int num_topics)
{
	FILE* fileptr;
	int d, k;
	fileptr = fopen(filename, "w");

	for (d = 0; d < num_docs; d++) {
		fprintf(fileptr, "%5.10f", gamma[d][0]);
		for (k = 1; k < num_topics; k++) {
			fprintf(fileptr, " %5.10f", gamma[d][k]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}


/*
* save the prediction results and the predictive R^2 value
*/
double MedLDA::save_prediction(char *filename, Corpus *corpus)
{
	double dmean = 0;
	double sumlikelihood = 0;
	int nterms = 0;
	double sumavglikelihood = 0;
	for ( int d=0; d<corpus->num_docs; d++ ) {
		//dmean += corpus->docs[d].responseVar / corpus->num_docs;
		sumlikelihood += corpus->docs[d].lhood;
		nterms += corpus->docs[d].total;
		sumavglikelihood += corpus->docs[d].lhood / corpus->docs[d].total;
	}
	double perwordlikelihood1 = sumlikelihood / nterms;
	double perwordlikelihood2 = sumavglikelihood / corpus->num_docs;

	int nAcc = 0;
	for ( int d=0; d<corpus->num_docs; d++ )
		if ( corpus->docs[d].gndlabel == corpus->docs[d].predlabel )
			nAcc += 1;
	double dAcc = (double)nAcc / corpus->num_docs;

	printf("Accuracy: %5.5f\n", dAcc);

	FILE* fileptr;
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "accuracy: %5.10f\n", dAcc );
	fprintf(fileptr, "perword likelihood1: %5.10f\n", perwordlikelihood1);
	fprintf(fileptr, "perword likelihood2: %5.10f\n", perwordlikelihood2);

	for (int d=0; d<corpus->num_docs; d++)
		fprintf(fileptr, "%d\t%d\n", corpus->docs[d].predlabel, corpus->docs[d].gndlabel);

	fclose(fileptr);

	return dAcc;
}

/*
* run_em
*/
int MedLDA::run_em(char* start, char* directory, Corpus* corpus, Params *param)
{
	m_dDeltaEll = param->DELTA_ELL;
	int d, n;
	double **var_gamma, **phi;

	// allocate variational parameters
	long runtime_start = get_runtime();

	var_gamma = (double**)malloc(sizeof(double*)*(corpus->num_docs));
	for (d = 0; d < corpus->num_docs; d++)
		var_gamma[d] = (double*)malloc(sizeof(double) * param->NTOPICS);

	int max_length = corpus->max_corpus_length();
	phi = (double**)malloc(sizeof(double*)*max_length);
	for (n = 0; n < max_length; n++)
		phi[n] = (double*)malloc(sizeof(double) * param->NTOPICS);

	// initialize model
	SuffStats* ss = NULL;
	if (strcmp(start, "seeded")==0) {
		new_model(corpus->num_docs, corpus->num_terms, param->NTOPICS, 
								param->NLABELS, param->INITIAL_C);
		ss = new_suffstats();
		corpus_init_ss(ss, corpus);
		mle(ss, param);
		for ( int k=0; k<m_nK; k++ ) 
			m_alpha[k] = param->INITIAL_ALPHA / param->NTOPICS;
	} else if (strcmp(start, "random")==0) {
		new_model(corpus->num_docs, corpus->num_terms, param->NTOPICS, 
								param->NLABELS, param->INITIAL_C);
		ss = new_suffstats();
		random_init_ss(ss, corpus);
		mle(ss, param);
		for ( int k=0; k<m_nK; k++ ) 
			m_alpha[k] = param->INITIAL_ALPHA / param->NTOPICS;
	} else {
		load_model(start);
		m_dC = param->INITIAL_C;
		ss = new_suffstats();

		ss->y = (int*) malloc(sizeof(int) * corpus->num_docs);
		ss->exp = (double**) malloc(sizeof(double*) * corpus->num_docs);
		for (int k=0; k<corpus->num_docs; k++ ) {
			ss->y[k] = corpus->docs[k].gndlabel;
			ss->exp[k] = (double*) malloc(sizeof(double)*param->NTOPICS);
		}
	}
	strcpy(ss->dir, directory);

	char filename[100];
	sprintf(filename, "%s/000",directory);
	save_model(filename);

	// run expectation maximization
	sprintf(filename, "%s/lhood.dat", directory);
	FILE* likelihood_file = fopen(filename, "w");

	int i = 0;
	double lhood, lhood_old = 0, converged = 1;
	int nIt = 0;
	while (((converged < 0) || (converged > param->EM_CONVERGED) 
		|| (i <= 2)) && (i <= param->EM_MAX_ITER))
	{
		printf("**** em iteration %d ****\n", i + 1);
		lhood = 0;
		zero_init_ss(ss);

		// e-step
		for (d = 0; d < corpus->num_docs; d++) {
			for (n = 0; n < max_length; n++) // initialize to uniform
				for ( int k=0; k<param->NTOPICS; k++ )
					phi[n][k] = 1.0 / (double) param->NTOPICS;

			if ((d % 1000) == 0) printf("Document %d\n",d);
			lhood += doc_e_step( &(corpus->docs[d]), var_gamma[d], phi, ss, param);
		}

		// m-step
		if ( mle(ss, param, false) ) {
			nIt = i + 1;
		} else {
			break;
		}

		// check for convergence
		lhood += m_dsvm_primalobj;

		converged = (lhood_old - lhood) / (lhood_old);
		lhood_old = lhood;

		// output model and lhood
		fprintf(likelihood_file, "%10.10f\t%5.5e\n", lhood, converged);
		fflush(likelihood_file);
		//if ((i % LAG) == 0)
		//{
		//	sprintf(filename,"%s\\%d",directory, i + 1);
		//	save_model(model, filename);
		//	sprintf(filename,"%s\\%d.gamma",directory, i + 1);
		//	save_gamma(filename, var_gamma, corpus->num_docs, m_nK);
		//}
		i ++;
	}
	long runtime_end = get_runtime();
	printf("Training time in (cpu-seconds): %.2f\n", ((float)runtime_end-(float)runtime_start)/100.0);

	// output the low-dimensional representation of data
	sprintf(filename, "MedLDA_(%dtopic)_train.txt", m_nK);

	//if ( param->NFOLDS == 5 ) partitionData(corpus, var_gamma, m_nK, m_nLabelNum);
	//else outputData(filename, corpus, var_gamma, m_nK, m_nLabelNum);
	/* outputData: not use! since the results are almost the same as outputData2. */
	//outputData2(filename, corpus, ss->exp, m_nK, m_nLabelNum);


	// output the final model
	sprintf(filename,"%s/final",directory);
	save_model(filename);
	sprintf(filename,"%s/final.gamma",directory);
	save_gamma(filename, var_gamma, corpus->num_docs, m_nK);


	// output the word assignments (for visualization)
	int nNum = 0, nAcc = 0;
	sprintf(filename, "%s/word-assignments.dat", directory);
	FILE* w_asgn_file = fopen(filename, "w");
	for (d = 0; d < corpus->num_docs; d++)
	{
		if ((d % 1000) == 0) printf("final e step Document %d\n",d);
		lhood += inference(&(corpus->docs[d]), d, var_gamma[d], phi, param);
		write_word_assignment(w_asgn_file, &(corpus->docs[d]), phi);

		//if ( d >= 856 ) {
			nNum ++;
			predict(&corpus->docs[d], phi);
			if ( corpus->docs[d].gndlabel == corpus->docs[d].predlabel ) nAcc ++;
		//}
	}
	fclose(w_asgn_file);
	fclose(likelihood_file);
	printf("\n\n MedLDA: double count accuracy: %.5f\n\n", (double)nAcc / nNum);


	for (d = 0; d < corpus->num_docs; d++)
		free(var_gamma[d]);
	free(var_gamma);
	for (n = 0; n < max_length; n++)
		free(phi[n]);
	free(phi);

	return nIt;
}

/*
* compute MLE lda model from sufficient statistics
*/
bool MedLDA::mle(SuffStats* ss, Params *param, bool bInit /*= true*/)
{
	int k; int w;

	// \beta parameters (K x N)
	for (k = 0; k < m_nK; k++) {
		for (w = 0; w < m_nNumTerms; w++) {
			if (ss->class_word[k][w] > 0) {
				m_dLogProbW[k][w] = log(ss->class_word[k][w]) - log(ss->class_total[k]);
			} else {
				m_dLogProbW[k][w] = -100;
			}
		}
	}

	// \alpha parameters
	if (!bInit && param->ESTIMATE_ALPHA == 1) {          // the same prior for all topics
		double alpha_suffstats = 0;
		for ( int k=0; k<m_nK; k++ ) {
			alpha_suffstats += ss->alpha_suffstats[k];
		}

		double alpha = opt_alpha(alpha_suffstats, ss->num_docs, m_nK);
		for ( int k=0; k<m_nK; k++ ) {
			m_alpha[k] = alpha;
		}
		printf("new alpha = %5.5f\n", alpha);
	} else if ( !bInit && param->ESTIMATE_ALPHA == 2 ) { // different priors for different topics
		double alpha_sum = 0;
		for ( int k=0; k<m_nK; k++ ) {
			alpha_sum += m_alpha[k];
		}

		for ( int k=0; k<m_nK; k++ ) {
			alpha_sum -= m_alpha[k];

			m_alpha[k] = opt_alpha(ss->alpha_suffstats[k], alpha_sum, ss->num_docs, m_nK);
		}
		printf("new alpha: ");
		for ( int k=0; k<m_nK; k++ ) {
			printf("%5.5f ", m_alpha[k]);
		}
		printf("\n");
	} else ;

	bool bRes = true;
	if ( !bInit ) {
		svmStructSolver(ss, param, m_dMu);
	} 

	return bRes;
}
void MedLDA::set_init_param(STRUCT_LEARN_PARM *struct_parm, LEARN_PARM *learn_parm, 
							KERNEL_PARM *kernel_parm, int *alg_type)
{
	/* set default */
	(*alg_type) = DEFAULT_ALG_TYPE;
	struct_parm->C = -0.01;
	struct_parm->slack_norm = 1;
	struct_parm->epsilon = DEFAULT_EPS;
	struct_parm->custom_argc = 0;
	struct_parm->loss_function = DEFAULT_LOSS_FCT;
	struct_parm->loss_type = DEFAULT_RESCALING;
	struct_parm->newconstretrain = 100;
	struct_parm->ccache_size = 5;
	struct_parm->batch_size = 100;
	struct_parm->delta_ell = m_dDeltaEll;

	strcpy(learn_parm->predfile, "trans_predictions");
	strcpy(learn_parm->alphafile, "");
	verbosity = 0;/*verbosity for svm_light*/
	struct_verbosity = 1; /*verbosity for struct learning portion*/
	learn_parm->biased_hyperplane = 1;
	learn_parm->remove_inconsistent = 0;
	learn_parm->skip_final_opt_check = 0;
	learn_parm->svm_maxqpsize = 10;
	learn_parm->svm_newvarsinqp = 0;
	learn_parm->svm_iter_to_shrink = -9999;
	learn_parm->maxiter = 100000;
	learn_parm->kernel_cache_size = 40;
	learn_parm->svm_c = 99999999;  /* overridden by struct_parm->C */
	learn_parm->eps = 0.001;       /* overridden by struct_parm->epsilon */
	learn_parm->transduction_posratio = -1.0;
	learn_parm->svm_costratio = 1.0;
	learn_parm->svm_costratio_unlab = 1.0;
	learn_parm->svm_unlabbound = 1E-5;
	learn_parm->epsilon_crit = 0.001;
	learn_parm->epsilon_a = 1E-10;  /* changed from 1e-15 */
	learn_parm->compute_loo = 0;
	learn_parm->rho = 1.0;
	learn_parm->xa_depth = 0;
	kernel_parm->kernel_type = 0;
	kernel_parm->poly_degree = 3;
	kernel_parm->rbf_gamma = 1.0;
	kernel_parm->coef_lin = 1;
	kernel_parm->coef_const = 1;
	strcpy(kernel_parm->custom,"empty");

	if(learn_parm->svm_iter_to_shrink == -9999) {
		learn_parm->svm_iter_to_shrink=100;
	}

	if((learn_parm->skip_final_opt_check) 
		&& (kernel_parm->kernel_type == LINEAR)) {
			printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
			learn_parm->skip_final_opt_check=0;
	}    
	if((learn_parm->skip_final_opt_check) 
		&& (learn_parm->remove_inconsistent)) {
			printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
			exit(0);
	}    
	if((learn_parm->svm_maxqpsize<2)) {
		printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
		exit(0);
	}
	if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
		printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
		printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
		exit(0);
	}
	if(learn_parm->svm_iter_to_shrink<1) {
		printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
		exit(0);
	}
	if(((*alg_type) < 0) || (((*alg_type) > 5) && ((*alg_type) != 9))) {
		printf("\nAlgorithm type must be either '0', '1', '2', '3', '4', or '9'!\n\n");
		exit(0);
	}
	if(learn_parm->transduction_posratio>1) {
		printf("\nThe fraction of unlabeled examples to classify as positives must\n");
		printf("be less than 1.0 !!!\n\n");
		exit(0);
	}
	if(learn_parm->svm_costratio<=0) {
		printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
		exit(0);
	}
	if(struct_parm->epsilon<=0) {
		printf("\nThe epsilon parameter must be greater than zero!\n\n");
		exit(0);
	}
	if((struct_parm->ccache_size<=0) && ((*alg_type) == 4)) {
		printf("\nThe cache size must be at least 1!\n\n");
		exit(0);
	}
	if(((struct_parm->batch_size<=0) || (struct_parm->batch_size>100))  
		&& ((*alg_type) == 4)) {
			printf("\nThe batch size must be in the interval ]0,100]!\n\n");
			exit(0);
	}
	if((struct_parm->slack_norm<1) || (struct_parm->slack_norm>2)) {
		printf("\nThe norm of the slacks must be either 1 (L1-norm) or 2 (L2-norm)!\n\n");
		exit(0);
	}
	if((struct_parm->loss_type != SLACK_RESCALING) 
		&& (struct_parm->loss_type != MARGIN_RESCALING)) {
			printf("\nThe loss type must be either 1 (slack rescaling) or 2 (margin rescaling)!\n\n");
			exit(0);
	}
	if(learn_parm->rho<0) {
		printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
		printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
		printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
		exit(0);
	}
	if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
		printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
		printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
		printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
		exit(0);
	}

	parse_struct_parameters(struct_parm);
}

void MedLDA::svmStructSolver(SuffStats* ss, Params *param, double *res)
{
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	STRUCT_LEARN_PARM struct_parm;
	STRUCTMODEL structmodel;
	int alg_type;

	/* set the parameters. */
	set_init_param(&struct_parm, &learn_parm, &kernel_parm, &alg_type);
	struct_parm.C = m_dC;

	// output the features
	char buff[512];
	sprintf(buff, "%s/Feature.txt", ss->dir);
	outputLowDimData(buff, ss);

	/* read the training examples */
	SAMPLE sample = read_struct_examples(buff, &struct_parm);

	if(param->SVM_ALGTYPE == 0)
		svm_learn_struct(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, NSLACK_ALG);
	//else if(alg_type == 1)
	//	svm_learn_struct(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, NSLACK_SHRINK_ALG);
	else if(param->SVM_ALGTYPE == 2) {
		struct_parm.C = m_dC * ss->num_docs;   // Note: in n-slack formulation, C is not divided by N.
		svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, ONESLACK_PRIMAL_ALG);
	}
	//else if(alg_type == 3)
	//	svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, ONESLACK_DUAL_ALG);
	//else if(alg_type == 4)
	//	svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, ONESLACK_DUAL_CACHE_ALG);
	//else if(alg_type == 9)
	//	svm_learn_struct_joint_custom(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel);
	else exit(1);

	/* get the optimal lagrangian multipliers. 
	*    Note: for 1-slack formulation: the "marginalization" is 
	*           needed for fast computation.
	*/
	int nVar = ss->num_docs * m_nLabelNum;
	for ( int k=0; k<nVar; k++ ) m_dMu[k] = 0;

	if ( param->SVM_ALGTYPE == 0 ) {
		for ( int k=1; k<structmodel.svm_model->sv_num; k++ ) {
			int n = structmodel.svm_model->supvec[k]->docnum;
			int docnum = structmodel.svm_model->supvec[k]->orgDocNum;
			m_dMu[docnum] = structmodel.svm_model->alpha[k];
		}
	} else if ( param->SVM_ALGTYPE == 2 ) {
		for ( int k=1; k<structmodel.svm_model->sv_num; k++ ) {
			int *vecLabel = structmodel.svm_model->supvec[k]->lvec;

			double dval = structmodel.svm_model->alpha[k] / ss->num_docs;
			for ( int d=0; d<ss->num_docs; d++ ) {
				int label = vecLabel[d];
				m_dMu[d*m_nLabelNum + label] += dval;
			}
		}
	} else ;

#ifdef _DEBUG
	FILE *fileptr = fopen("MuSolution.txt", "a");
	for ( int i=0; i<ss->num_docs; i++ ) {
		for ( int k=0; k<m_nLabelNum; k++ ) {
			int muIx = i * m_nLabelNum + k;
			if ( m_dMu[muIx] > 0 ) fprintf(fileptr, "%d:%.5f ", k, m_dMu[muIx]);
		}
		fprintf(fileptr, "\n");
	}
	fprintf(fileptr, "\n\n");
	fclose(fileptr);
#endif


	//FILE *fileptr = fopen("SVMLightSolution.txt", "a");
	// set the SVM parameters.
	m_dB = structmodel.svm_model->b;
	for ( int y=0; y<m_nLabelNum; y++ ) {
		for ( int k=0; k<m_nK; k++ ){
			int etaIx = y * m_nK + k;
			m_dEta[etaIx] = structmodel.w[etaIx+1];
			//fprintf(fileptr, "%5.5f:%5.5f ", m_dEta[etaIx], structmodel.w[etaIx+1]);
		}
		//fprintf(fileptr, "\n");
	}
	//fprintf(fileptr, "\n%5.5f\n\n", m_dB);
	//fclose(fileptr);
	m_dsvm_primalobj = structmodel.primalobj;

	// free the memory
	free_struct_sample(sample);
	free_struct_model(structmodel);
}
void MedLDA::outputLowDimData(char *filename, SuffStats *ss)
{
	FILE *fileptr = fopen(filename, "w");
	for ( int d=0; d<ss->num_docs; d++ ) {
		int label = ss->y[d];

		fprintf(fileptr, "%d %d", m_nK, label);

		for ( int k=0; k<m_nK; k++ ) {
			fprintf(fileptr, " %d:%.10f", k, ss->exp[d][k]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}

/*
* allocate sufficient statistics
*/
SuffStats* MedLDA::new_suffstats( )
{
	int num_topics = m_nK;
	int num_terms = m_nNumTerms;

	SuffStats* ss = (SuffStats*)malloc(sizeof(SuffStats));
	ss->class_total = (double*)malloc(sizeof(double)*num_topics);
	ss->class_word = (double**)malloc(sizeof(double*)*num_topics);
	for (int i=0; i<num_topics; i++) {
		ss->class_total[i] = 0;
		ss->class_word[i] = (double*)malloc(sizeof(double)*num_terms);
		for (int j=0; j<num_terms; j++) {
			ss->class_word[i][j] = 0;
		}
	}

	ss->alpha_suffstats = (double*)malloc(sizeof(double)*m_nK);
	memset(ss->alpha_suffstats, 0, sizeof(double)*m_nK);

	return(ss);
}


/*
* various intializations for the sufficient statistics
*/
void MedLDA::zero_init_ss(SuffStats* ss)
{
	for (int k=0; k<m_nK; k++) {
		ss->class_total[k] = 0;
		for (int w=0; w<m_nNumTerms; w++) {
			ss->class_word[k][w] = 0;
		}
	}
	ss->num_docs = 0;
	memset(ss->alpha_suffstats, 0, sizeof(double)*m_nK);
}

void MedLDA::random_init_ss(SuffStats* ss, Corpus* c)
{
	int num_topics = m_nK;
	int num_terms = m_nNumTerms;
	for (int k = 0; k < num_topics; k++) {
		for (int n = 0; n < num_terms; n++) {
			ss->class_word[k][n] += 10.0 /*1.0/num_terms*/ + myrand();
			ss->class_total[k] += ss->class_word[k][n];
		}
	}

	ss->y = (int*) malloc(sizeof(int) * c->num_docs);
	ss->exp = (double**) malloc(sizeof(double*) * c->num_docs);
	for (int k=0; k<c->num_docs; k++ ) {
		ss->y[k] = c->docs[k].gndlabel;
		ss->exp[k] = (double*) malloc(sizeof(double)*m_nK);
	}
}

void MedLDA::corpus_init_ss(SuffStats* ss, Corpus* c)
{
	int num_topics = m_nK;
	int i, k, d, n;
	Document* doc;

	for (k = 0; k < num_topics; k++) {
		for (i = 0; i < NUM_INIT; i++) {
			d = floor(myrand() * c->num_docs);
			printf("initialized with Document %d\n", d);
			doc = &(c->docs[d]);
			for (n = 0; n < doc->length; n++) {
				ss->class_word[k][doc->words[n]] += doc->counts[n];
			}
		}
		for (n = 0; n < m_nNumTerms; n++) {
			ss->class_word[k][n] += 1.0;
			ss->class_total[k] = ss->class_total[k] + ss->class_word[k][n];
		}
	}

	// for sLDA only
	ss->y = (int*) malloc(sizeof(int) * c->num_docs);
	ss->exp = (double**) malloc(sizeof(double*) * c->num_docs);
	for ( d=0; d<c->num_docs; d++ ) {
		ss->y[d] = c->docs[d].gndlabel;
		ss->exp[d] = (double*) malloc(sizeof(double)*num_topics);
	}
}

/*
* allocate new model
*/
void MedLDA::new_model(int num_docs, int num_terms, int num_topics, int num_labels, double C)
{
	int i,j;
	m_nK = num_topics;
	m_nLabelNum = num_labels;
	m_nNumTerms = num_terms;

	m_alpha = (double*)malloc(sizeof(double) * m_nK);
	for ( int k=0; k<m_nK; k++ ) m_alpha[k] = 1.0 / num_topics;
	m_dLogProbW = (double**)malloc(sizeof(double*)*num_topics);
	m_dEta = (double*)malloc(sizeof(double) * num_topics * num_labels);
	m_dMu = (double*)malloc(sizeof(double) * num_docs * num_labels);
	for (i = 0; i < num_topics; i++)
	{
		m_dLogProbW[i] = (double*)malloc(sizeof(double)*num_terms);
		for (j = 0; j < num_terms; j++) m_dLogProbW[i][j] = 0;
		for (j = 0; j < num_labels; j++) m_dEta[i*num_labels + j] = 0;
	}
	for (i = 0; i < num_docs; i ++)
		for (j = 0; j < num_labels; j++)
			m_dMu[i*num_labels + j] = 0;

	m_nDim = num_docs;
	m_dC = C;
}

void MedLDA::free_model()
{
	if ( m_dLogProbW != NULL ) {
		for (int i=0; i<m_nK; i++) {
			free(m_dLogProbW[i]);
		}
		free(m_dLogProbW);
	}
	if ( m_dEta != NULL ) free(m_dEta);
	if ( m_dMu != NULL )  free(m_dMu);
	if ( m_alpha != NULL ) free(m_alpha);
}


/*
* save an MedLDA model
*/
void MedLDA::save_model(char* model_root)
{
	char filename[100];
	FILE* fileptr;
	int i, j;

	sprintf(filename, "%s.beta", model_root);
	fileptr = fopen(filename, "w");

	fprintf(fileptr, "%5.10f\n", m_dB);

	for (i = 0; i < m_nK; i++) {
		// the first element is eta[k]
		for ( int k=0; k<m_nLabelNum; k++ ) {
			if ( k == m_nLabelNum-1 )
				fprintf(fileptr, "%5.10f", m_dEta[i+k*m_nK]);
			else 
				fprintf(fileptr, "%5.10f ", m_dEta[i+k*m_nK]);
		}

		for (j = 0; j < m_nNumTerms; j++) {
			fprintf(fileptr, " %5.10f", m_dLogProbW[i][j]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	sprintf(filename, "%s.other", model_root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "num_topics %d\n", m_nK);
	fprintf(fileptr, "num_labels %d\n", m_nLabelNum);
	fprintf(fileptr, "num_terms %d\n", m_nNumTerms);
	fprintf(fileptr, "num_docs %d\n", m_nDim);
	fprintf(fileptr, "alpha ");
	for ( int k=0; k<m_nK; k++ ) {
		fprintf(fileptr, "%5.10f ", m_alpha[k]);
	}
	fprintf(fileptr, "\n");
	fprintf(fileptr, "C %5.10f\n", m_dC);
	fclose(fileptr);
}

void MedLDA::load_model(char* model_root)
{
	char filename[100];
	FILE* fileptr;
	int i, j, num_terms, num_topics, num_labels, num_docs;
	float x, alpha, C, learnRate;
	vector<double> vecAlpha;

	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_labels %d\n", &num_labels);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fscanf(fileptr, "alpha ");
	for ( int k=0; k<num_topics; k++ ) {
		if ( k == num_topics - 1 ) fscanf(fileptr, "%f \n", &alpha);
		else fscanf(fileptr, "%f ", &alpha);
		vecAlpha.push_back(alpha);
	}
	fscanf(fileptr, "C %f\n", &C);
	fclose(fileptr);

	new_model(num_docs, num_terms, num_topics, num_labels, C);
	for ( int k=0; k<num_topics; k++ ) {
		m_alpha[k] = vecAlpha[k];
	}

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");

	fscanf(fileptr, "%lf\n", &m_dB);

	for ( int i=0; i<m_nK; i++) {
		for ( int k=0; k<m_nLabelNum; k++ ) {
			fscanf(fileptr, "%f", &x);
			m_dEta[i+k*m_nK] = x;
		}

		for ( int j=0; j<m_nNumTerms; j++) {
			fscanf(fileptr, "%f", &x);
			m_dLogProbW[i][j] = x;
		}
	}

	fclose(fileptr);
}

// save the low-dimentional data for 5 fold cv
void MedLDA::partitionData(Corpus *corpus, double** gamma, int ntopic, int nLabels)
{
	FILE *fileptr = fopen("randomorder.txt", "r");
	char buff[512];
	vector<double> order;
	while ( fscanf(fileptr, "%d", buff) != EOF ) {
		order.push_back( atof(buff) );
	}
	fclose(fileptr);

	// partition into 5 parts for 5 fold cv
	int nunit = corpus->num_docs / 5;
	for ( int k=1; k<=5; k++ ) {
		sprintf(buff, "MedLDA_train_(%dtopic)_cv5(%d).txt", ntopic, k);
		FILE *fileptr1 = fopen(buff, "w");
		sprintf(buff, "MedLDA_test_(%dtopic)_cv5(%d).txt", ntopic, k);
		FILE *fileptr2 = fopen(buff, "w");
		for ( int i=0; i<corpus->num_docs; i++ ) {
			int ndocIx = order[i];

			int label = corpus->docs[ndocIx].gndlabel;
			if ( nLabels == 2 ) {
				if ( corpus->docs[ndocIx].gndlabel == -1 ) label = 1;
				if ( corpus->docs[ndocIx].gndlabel == 1 ) label = 0;
			}

			bool btrain = true;
			if ( k < 5 && (i>=(k-1)*nunit) && (i<k*nunit) ) btrain = false;
			else if ( k == 5 && (i >= (k-1) * nunit) ) btrain = false;

			double dNorm = 0;
			for ( int k=0; k<ntopic; k++ ) dNorm += gamma[ndocIx][k];

			if ( btrain ) {
				fprintf(fileptr1, "%d %d", ntopic, label);
				for ( int k=0; k<ntopic; k++ ) 
					fprintf(fileptr1, " %d:%.10f", k, gamma[ndocIx][k] / dNorm);
				fprintf(fileptr1, "\n");
			} else {
				fprintf(fileptr2, "%d %d", ntopic, label);
				for ( int k=0; k<ntopic; k++ ) 
					fprintf(fileptr2, " %d:%.10f", k, gamma[ndocIx][k] / dNorm);
				fprintf(fileptr2, "\n");
			}
		}
		fclose(fileptr1);
		fclose(fileptr2);
	}
}

// save the low-dimentional data
void MedLDA::outputData(char *filename, Corpus *corpus, double** gamma, int ntopic, int nLabels)
{
	FILE *fileptr = fopen(filename, "w");
	for ( int i=0; i<corpus->num_docs; i++ ) {
		int label = corpus->docs[i].gndlabel;
		if ( nLabels == 2 ) {
			if ( corpus->docs[i].gndlabel == -1 ) label = 1;
			if ( corpus->docs[i].gndlabel == 1 ) label = 0;
		}

		double dNorm = 0;
		for ( int k=0; k<ntopic; k++ ) dNorm += gamma[i][k];

		fprintf(fileptr, "%d %d", ntopic, label);
		for ( int k=0; k<ntopic; k++ ) 
			fprintf(fileptr, " %d:%.10f", k, gamma[i][k] / dNorm);
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}

void MedLDA::outputData2(char *filename, Corpus *corpus, double **exp, int ntopic, int nLabels)
{
	FILE *fileptr = fopen(filename, "w");

	for ( int i=0; i<corpus->num_docs; i++ ) {
		int label = corpus->docs[i].gndlabel;
		if ( nLabels == 2 ) {
			if ( corpus->docs[i].gndlabel == -1 ) label = 1;
			if ( corpus->docs[i].gndlabel == 1 ) label = 0;
		}

		fprintf(fileptr, "%d %d", ntopic, label);
		for ( int k=0; k<ntopic; k++ ) 
			fprintf(fileptr, " %d:%.10f", k, exp[i][k]);
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}


/*
* inference only
*/
double MedLDA::infer(char* model_dir, Corpus* corpus, Params *param)
{
	FILE* fileptr;
	char filename[100];
	int i, d, n;
	double **var_gamma, lhood, **phi;
	Document* doc;

	char model_root[512];
	sprintf(model_root, "%s/final", model_dir);
	load_model(model_root);

	// remove unseen words
	if ( corpus->num_terms > m_nNumTerms ) {
		for ( int i=0; i<corpus->num_docs; i ++ )
		{
			for ( int k=0; k<corpus->docs[i].length; k++ )
				if ( corpus->docs[i].words[k] >= m_nNumTerms )
					corpus->docs[i].words[k] = m_nNumTerms - 1;
		}
	}

	var_gamma = (double**)malloc(sizeof(double*)*(corpus->num_docs));
	double **exp = (double**)malloc(sizeof(double*)*corpus->num_docs);
	for (i = 0; i < corpus->num_docs; i++) {
		var_gamma[i] = (double*)malloc(sizeof(double)*m_nK);
		exp[i] = (double*)malloc(sizeof(double) * m_nK);
	}
	
	sprintf(filename, "%s/evl-lda-lhood.dat", model_dir);
	fileptr = fopen(filename, "w");
	for (d = 0; d < corpus->num_docs; d++)
	{
		if (((d % 1000) == 0) && (d>0)) printf("Document %d\n",d);

		doc = &(corpus->docs[d]);
		phi = (double**) malloc(sizeof(double*) * doc->length);
		for (n = 0; n < doc->length; n++)
		{
			phi[n] = (double*) malloc(sizeof(double) * m_nK);
			// initialize to uniform distrubtion
			for ( int k=0; k<m_nK; k++ )
				phi[n][k] = 1.0 / (double) m_nK;
		}

		lhood = inference_pred(doc, var_gamma[d], phi, param);

		// do prediction
		predict(doc, phi);
		doc->lhood = lhood;
		fprintf(fileptr, "%5.5f\n", lhood);

		// update the exp
		for (int k = 0; k < m_nK; k++) {
			double dVal = 0;
			for (n = 0; n < doc->length; n++) {
				dVal += phi[n][k] * doc->counts[n] / doc->total;
			}
			exp[d][k] = dVal;
		}
	}
	fclose(fileptr);

	// output the predicted representation of MedLDA
	sprintf(filename, "MedLDA_(%dtopic)_test.txt", m_nK);
	outputData2(filename, corpus, exp, m_nK, m_nLabelNum);

	sprintf(filename, "%s/evl-gamma.dat", model_dir);
	save_gamma(filename, var_gamma, corpus->num_docs, m_nK);

	// save the prediction performance
	sprintf(filename, "%s/evl-performance.dat", model_dir);
	double dAcc = save_prediction(filename, corpus);

	fileptr = fopen("overall-res.txt", "a");
	fprintf(fileptr, "setup (K: %d; C: %.3f; fold: %d; ell: %.2f; dual-opt: %d; alpha: %d; svm_alg: %d; maxIt: %d): accuracy %.3f\n", 
		m_nK, m_dC, param->NFOLDS, param->DELTA_ELL, param->PHI_DUALOPT, param->ESTIMATE_ALPHA, param->SVM_ALGTYPE, param->EM_MAX_ITER, dAcc);
	fclose(fileptr);

	// free memory
	for ( i=0; i<corpus->num_docs; i++ ) {
		free( var_gamma[i] );
		free( exp[i] );
	}
	free( var_gamma );
	free( exp );

	return dAcc;
}

void MedLDA::predict(Document *doc, double **phi)
{
	doc->predlabel = -1;
	double dMaxScore = 0;
	for ( int y=0; y<m_nLabelNum; y++ )
	{
		double dScore = 0;
		for ( int k=0; k<m_nK; k++ ) {
			int etaIx = y * m_nK + k;

			double dVal = 0;
			for ( int n=0; n<doc->length; n++ )
				dVal += phi[n][k] * doc->counts[n] / doc->total;

			dScore += dVal * m_dEta[etaIx];
		}
		dScore -= m_dB;

		if ( doc->predlabel == -1 || dScore > dMaxScore ) {
			doc->predlabel = y;
			dMaxScore = dScore;
		}
	}
}