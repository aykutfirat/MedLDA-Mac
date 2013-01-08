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

#include "Corpus.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

Corpus::Corpus(void)
{
	docs = NULL;
}

Corpus::~Corpus(void)
{
	if (docs != NULL ) free(docs);
}


void Corpus::shuffle()
{
	srand(time(NULL));
	int n = 0;
	for ( n=0; n<num_docs*100; n++ )
	{
		int ix1 = rand() % num_docs;
		int ix2 = rand() % num_docs;
		if ( ix1 == ix2 ) continue;
		
		Document p = docs[ix1];
		docs[ix1] = docs[ix2];
		docs[ix2] = p;
	}
}

Corpus* Corpus::get_traindata(const int&nfold, const int &foldix)
{
	int nunit = num_docs / nfold;

	Corpus *subc = new Corpus();
	subc->docs = 0;
	subc->num_docs = 0;
	subc->num_terms = 0;
	int nd = 0, nw = 0;
	for ( int i=0; i<num_docs; i++ )
	{
		if ( foldix < nfold ) {
			if ( (i >= (foldix-1)*nunit) && ( i < foldix*nunit ) ) continue;
		} else {
			if ( i >= (foldix-1) * nunit ) continue;
		}

		subc->docs = (Document*) realloc(subc->docs, sizeof(Document)*(nd+1));
		subc->docs[nd].length = docs[i].length;
		subc->docs[nd].total = docs[i].total;
		subc->docs[nd].words = (int*)malloc(sizeof(int)*docs[i].length);
		subc->docs[nd].counts = (int*)malloc(sizeof(int)*docs[i].length);
		
		// read the response variable
		subc->docs[nd].gndlabel = docs[i].gndlabel;

		for (int n=0; n<docs[i].length; n++) {
			subc->docs[nd].words[n] = docs[i].words[n];
			subc->docs[nd].counts[n] = docs[i].counts[n];
			if (docs[i].words[n] >= nw) { nw = docs[i].words[n] + 1; }
		}
		nd++;
	}
	subc->num_docs = nd;
	subc->num_terms = nw;
	return subc;
}

Corpus* Corpus::get_testdata(const int&nfold, const int &foldix)
{
	int nunit = num_docs / nfold;

	Corpus *subc = new Corpus();
	subc->docs = 0;
	subc->num_docs = 0;
	subc->num_terms = 0;
	int nd = 0, nw = 0;
	for ( int i=0; i<num_docs; i++ )
	{
		if ( foldix < nfold ) {
			if ( i < ((foldix-1)*nunit) || i >= foldix*nunit ) continue;
		} else {
			if ( i < (foldix-1) * nunit ) continue;
		}

		subc->docs = (Document*) realloc(subc->docs, sizeof(Document)*(nd+1));
		subc->docs[nd].length = docs[i].length;
		subc->docs[nd].total = docs[i].total;
		subc->docs[nd].words = (int*)malloc(sizeof(int)*docs[i].length);
		subc->docs[nd].counts = (int*)malloc(sizeof(int)*docs[i].length);
		
		// read the response variable
		subc->docs[nd].gndlabel = docs[i].gndlabel;

		for (int n = 0; n < docs[i].length; n++)
		{
			subc->docs[nd].words[n] = docs[i].words[n];
			subc->docs[nd].counts[n] = docs[i].counts[n];
			if (docs[i].words[n] >= nw) { nw = docs[i].words[n] + 1; }
		}
		nd++;
	}
	subc->num_docs = nd;
	subc->num_terms = nw;
	return subc;
}

void Corpus::reorder(char *filename)
{
	int num, ix=0;
	int *order = (int*)malloc(sizeof(int)*num_docs);
	FILE *fileptr = fopen(filename, "r");
	while ( (fscanf(fileptr, "%10d", &num) != EOF ) ) {
		order[ix] = num;
		ix ++;
	}
	
	Document *tmp_docs = (Document*)malloc(sizeof(Document) * num_docs);
	for ( int i=0; i<num_docs; i++ )
		tmp_docs[i] = docs[i];
	for ( int i=0; i<num_docs; i++ )
		docs[i] = tmp_docs[order[i]];
	free(tmp_docs);
	free(order);
}

void Corpus::read_data(char* data_filename, int nLabels)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw;

	printf("reading data from %s\n", data_filename);
	docs = 0;
	num_terms = 0;
	num_docs = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d", &length) != EOF))
	{
		docs = (Document*) realloc(docs, sizeof(Document)*(nd+1));
		docs[nd].length = length;
		docs[nd].total = 0;
		docs[nd].words = (int*)malloc(sizeof(int)*length);
		docs[nd].counts = (int*)malloc(sizeof(int)*length);
		
		int label;
		fscanf(fileptr, "%d", &label);
		docs[nd].gndlabel = label;

		for (n = 0; n < length; n++) {
			fscanf(fileptr, "%10d:%10d", &word, &count);
			word = word - OFFSET;
			docs[nd].words[n] = word;
			docs[nd].counts[n] = count;
			docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
		}
		nd++;
	}
	fclose(fileptr);
	num_docs = nd;
	num_terms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);
}

int Corpus::max_corpus_length( )
{
	int max = 0;
	for (int n=0; n<num_docs; n++)
		if (docs[n].length > max) max = docs[n].length;
	return(max);
}
