#pragma once

#define OFFSET 0;                  // offset for reading data

typedef struct
{
	int gndlabel;	// the ground truth response variable value
	int predlabel;	// the predicted response variable value
	int lossAugLabel;
	double lhood;
    int* words;
    int* counts;
    int length;
    int total;
} Document;


class Corpus
{
public:
	Corpus(void);
public:
	~Corpus(void);

	void read_data(char* data_filename, int nLabels);
	Corpus* get_traindata(const int&nfold, const int &foldix);
	Corpus* get_testdata(const int&nfold, const int &foldix);
	void reorder(char *filename);

	int max_corpus_length( );

	void shuffle();

public:
    Document* docs;
    int num_terms;
    int num_docs;
};
