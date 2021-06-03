
### New Readme


### pip libraries:
nltk
pyenchant
compound-word-splitter
textwiser
jamspell
wikipedia2vec
genism
pymed

### apt pacakages:
libenchant1c2a

### downloads
Download and extract en.tar.gz from https://github.com/bakwc/JamSpell#download-models in project directory
If you wnat to run v1(not part of final analysis) - download and extract enwiki_20180420 (window=5, iteration=10, negative=15): 100d (bin) from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/ in project directory
Rest of the downloads are automatically done upon requirement

### Running
A new evaluate function is used in main.py file (line 365) this is set to get best results (i.e. v5 as mentioned in the document)
A general python3 main.py will do the work of plotting (file name has v7 in it) and printing
Different versions of the project can be tried out by uncommenting functions (main.py - line 283-286) and variables(informationRetrival.py line 11-59, line 150-200)

### Results
A complete set of results for different versions is provided in file results.txt
The plots for different versions are present in Plots/ dorectory and plots for new tryouts also appear in the same folder










### Readme for old version

This folder contains the additional files required for Part 2 of the assignment, involving building a search engine application. Note that this code works for both Python 2 and Python 3. 

The following files have been added:
informationRetrieval.py and evaluation.py - Implement the corresponding tasks inside the functions in these files.

The following file has been updated:
main.py - The main module that contains the outline of the Search Engine. It has been updated to include calls to the information retrieval and evaluation tasks, in addition to the tasks solved in Part 1 of the assignment. Do not change anything in this file.

For this part of the assignment, you are advised to make a copy of the completed code from Part 1 of the assignment - replace the main file with the updated version and add and fill in the new files (informationRetrieval.py and evaluation.py).

To test your code, run main.py as before with the appropriate arguments.
Usage: main.py [-custom] [-dataset DATASET FOLDER] [-out_folder OUTPUT FOLDER]
               [-segmenter SEGMENTER TYPE (naive|punkt)] [-tokenizer TOKENIZER TYPE (naive|ptb)] 

When the -custom flag is passed, the system will take a query from the user as input. For example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will print the IDs of the five most relevant documents to the query to standard output.

When the flag is not passed, all the queries in the Cranfield dataset are considered and precision@k, recall@k, f-score@k, nDCG@k and the Mean Average Precision are computed.

In both the cases, *queries.txt files and *docs.txt files will be generated in the OUTPUT FOLDER after each stage of preprocessing of the documents and queries.
