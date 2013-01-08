# Makefile for MedLDA 1/5/2013 
# Original code by Jun Zhu junzhu [at] cs.cmu.edu

all: medlda

clear:
	rm -f ./SVMLight/*.o; 
	rm -f ./SVM_Multiclass/*.o;
	rm -f *.o *.exe MedLDA;
	
svm_light_clean: 
	cd svmlight; g++ -c *.cpp

svm_struct_clean: 
	cd svm_multiclass; g++ -c *.cpp

medlda_clean: 
	g++ -c *.cpp

medlda: svm_light_clean svm_struct_clean medlda_clean
	g++ ./SVMLight/*.o ./SVM_Multiclass/*.o *.o -o MedLDA
