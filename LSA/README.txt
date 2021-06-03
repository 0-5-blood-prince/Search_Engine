Author for LSA models: Mooizz 

This folder contains the files which are required to building a search engine application with Latent Semantic Analysis Methods. Note that this code works for both Python 2 and Python 3. 
LSI.py file contains the main function and all the code for preprocessing, implementing all models and evaluating these models to test hypothesis.

informationRetrieval.py contains the functions required to Implement the different models (Basic Vector Space Model, 
Latent Semantic Indexing, Supervised Learning Model which implements relevance feedback)

evaluation.py contains all the functions required for computing evaluation metrics and some other helper functions

tokenization.py, sentenceSegmentation.py, inflectionReduction.py, stopwordRemoval.py files contain the functions required for Preprocessing the documents and queries

To test the code, run LSI.py as before with the appropriate arguments.
Usage: LSI.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 
                [-k number of factors for SVD] [-num_retrieved (Number of first ranked documents to consider for evaluation)]
                [-method (hyp1(comparing LSI vs Basic VSM)|hyp2(comparing LSI vs Supervised Model)|lsi|basic)]
                [-experiment (whether to experiment or not [=1 only for methods hyp1 and hyp2 ])]

The report contains the experimentation and results of evaluating the models
hyp1  : This is to test the hypothesis that LSI model is better than basic Vector Space Model 
To replicate the results do 
    Experiment: python LSI.py -experiment 1 -method hyp1 -num_retrieved 20
    Final Result : python LSI.py -method hyp1 -num_retrieved 20 -k [num factors|default 200 for which the performance was better]
hyp2  : This is to test the hypothesis that Supervised Learning model which includes relevance feedback is better than LSI model.
To replicate the results do 
    Experiment: python LSI.py -experiment 1 -method hyp2 -num_retrieved 20
    Final Result : python LSI.py -method hyp2 -num_retrieved 20 
Plots are saved in output folder

When the -custom flag is passed, the system will take a query from the user as input.
 but only when -method flag is lsi or basic 
 For example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will print the IDs of the five most relevant documents to the query to standard output.

When the flag is not passed, all the queries in the Cranfield dataset are considered and evaluations are done on the specified models.

In both the cases, *queries.txt files and *docs.txt files will be generated in the OUTPUT FOLDER after each stage of preprocessing of the documents and queries.
