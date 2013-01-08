***************************
MedLDA: Max-margin Supervised Topic Models
***************************

Jun Zhu
junzhu[at]cs.cmu.edu

(C) Copyright 2010, Jun Zhu (junzhu [at] cs [dot] cmu [dot] edu)

This file is part of MedLDA.

MedLDA is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your
option) any later version.

MedLDA is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA

------------------------------------------------------------------------

This is a C implementation of max-margin supervised topic model (MedLDA), a
model of discrete data which is fully described in Zhu et al. (2010)
(http://www.cs.cmu.edu/~junzhu/MedLDAc/MedLDA_draft.pdf).

------------------------------------------------------------------------


TABLE OF CONTENTS


A. COMPILING

B. TOPIC ESTIMATION

   1. SETTINGS FILE

   2. DATA FILE FORMAT

C. INFERENCE

D. ESTIMATION AND INFERENCE

E. QUESTIONS, COMMENTS, PROBLEMS, UPDATE ANNOUNCEMENTS


------------------------------------------------------------------------

A. COMPILING

   1. For Windows users:
	Use Visual Studio 2005 to open "MedLDAc.sln". Set the "boost" 
library (http://www.boost.org/) correctly and compile.
   2. For Linux users:
	g++ *.cpp svmlight/*.cpp svm_multiclass/*.cpp -o medlda -lm
	or use make

------------------------------------------------------------------------

B. TOPIC ESTIMATION

Estimate the model by executing:

     MEDsLDAc est [k] [labels] [fold] [initial C] [l] [dir root] [random/seeded/*]

The term [random/seeded/*] > describes how the topics will be
initialized.  "random" initializes each topic randomly; "seeded"
initializes each topic to a distribution smoothed from a randomly
chosen document; or, you can specify a model name to load a
pre-existing model as the initial model (this is useful to continue EM
from where it left off).  The data used for estimation is specified in the 
Settings file, as explained below.

The model (i.e., \alpha and \beta_{1:K}) and variational posterior
Dirichlet parameters will be saved in a directory specified by "dir root", and
the directoy is of the form "<dir root><k>_c<initial C>_f<fold>".
Additionally, there will be a log file for the likelihood bound and convergence score 
at each iteration.  The algorithm runs until that score is less than "em_convergence" (from
the settings file) or "em_max_iter" iterations are reached.

The saved models are in two files:

     final.other contains alpha.

     final.beta contains the log of the topic distributions.
     Each line is a topic; in line k, each entry is log p(w | z=k)

The variational posterior Dirichlets are in:

     final.gamma

The settings file and data format are described below.


1. Settings file

See settings.txt for a sample. These are placeholder values; they
should be experimented with.

This is of the following form:

     var max iter [integer e.g., 10 or -1]
     var convergence [float e.g., 1e-8]
     em max iter [integer e.g., 100]
     em convergence [float e.g., 1e-5]
     model C [positive float e.g., 16.0]
     init alpha [float e.g., 0.1]
     svm_alg_type [0/2]
     alpha [0/1/2]
     inner-cv [true/false]
     inner_foldnum [integer e.g., 5]
     cv_paramnum [integer e.g., 7]
     [candidate C value, e.g., 1.0]
     [candidate C value, e.g., 4.0]
     [candidate C value, e.g., 9.0]
     [candidate C value, e.g., 16.0]
     [candidate C value, e.g., 25.0]
     [candidate C value, e.g., 36.0]
     [candidate C value, e.g., 49.0]
     train_file: [string e.g., ..\train.dat]
     test_file: [string e.g., ..\test.dat]

where the settings are

     [var max iter]

     The maximum number of iterations of coordinate ascent variational
     inference for a single document.  A value of -1 indicates "full"
     variational inference, until the variational convergence
     criterion is met.

     [var convergence]

     The convergence criteria for variational inference.  Stop if
     (score_old - score) / abs(score_old) is less than this value (or
     after the maximum number of iterations).  Note that the score is
     the lower bound on the likelihood for a particular document.

     [em max iter]

     The maximum number of iterations of variational EM.

     [em convergence]

     The convergence criteria for varitional EM.  Stop if (score_old -
     score) / abs(score_old) is less than this value (or after the
     maximum number of iterations).  Note that "score" is the lower
     bound on the likelihood for the whole corpus.

     [svm_alg_type]
     
     If set to [0] then the n-slack multi-class SVM is used. If set to [2],
     then the 1-slack multi-class SVM is used. In our testing, the 1-slack
     SVM is more faster.
     
     [alpha]

     If set to [0] then alpha does not change from iteration to
     iteration.  If set to [1], then alpha is estimated along
     with the topic distributions.  If set to [2], then k different
     alpha (one for each topic) is estimated along with the topic distributions.
     
     [inner-cv]
     
     If set to [true], then cross-validation is used during training to select C
     from a list of candidates specified after [cv_paramnum]. If set to [false],
     the regularization constant C is set as the initial value [model C].
     
     [inner_foldnum]
     
     The number of folds for inner cross validation during training.
     
     [train_file]
     
     The file name of training data.
     
     [test_file]
     
     The file name of testing data.


2. Data format

Under MEDsLDAc, the words of each document are assumed exchangeable.  Thus,
each document is succinctly represented as a sparse vector of word
counts. The data is a file where each line is of the form:

     [M] [label] [term_1]:[count] [term_2]:[count] ...  [term_M]:[count]

where [M] is the number of unique terms in the document; [label] is the true label
of the document; and the [count] associated with each term is how many times that 
term appeared in the document.  Note that [term_1] is an integer which indexes the
term; it is not a string.


------------------------------------------------------------------------

C. INFERENCE

To perform inference on a different set of data (in the same format as
for estimation), execute:

     MEDsLDAc inf [labels] [model]

Variational inference is performed on the data using the model in
[model].* (see above).  Three files will be created : evl-gamma.dat are
the variational Dirichlet parameters for each document;
evl-lda-lhood.dat is the bound on the likelihood for each document;
and evl-performance.dat is the classification accuracy and detailed
labeling results for each document. 


------------------------------------------------------------------------

D. ESTIMATION AND INFERENCE

For simplicity, a command is provided for doing both estimation and inference.  
Usage is:

     MEDsLDAc estinf [k] [labels] [fold] [initial C] [l] [random/seeded/*]
	 Example: ./MedLDA estinf 40 20 4 1 3600 random

------------------------------------------------------------------------

E. QUESTIONS, COMMENTS, PROBLEMS, AND UPDATE ANNOUNCEMENTS

Questions, comments, and problems should be addressed to,
junzhu@cs.cmu.edu.

Update announcements will be posted at: http://cs.cmu.edu/~junzhu/medlda.htm