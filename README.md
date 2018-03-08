# Introduction

This is my submission to the [PAN16](http://pan.webis.de/clef16/pan16-web/author-profiling.html) Author Profiling
competition, which analyzes tweets, blogs and hotel reviews and predict the sex and age of the person who wrote them.
The model works for English, Spanish and Dutch texts. This submission won 3rd place at the competition.

Disclaimer: this is my old code from 2016 and I haven't looked into it lately. In the future I would like to port it to 
Python 3. Currently it has some neat scikit-learn trickery including an example on a complex model
pipeline, modifications of the TF-IDF Vectorizer and much more. It also includes a lot of interesting advanced NLP
features about which you can read in my [Thesis](./Thesis+Papers/IvanBilan_CrossGenreAuthorProfiling_2016.pdf) or the
[conference paper](http://ceur-ws.org/Vol-1609/16090824.pdf). Since
this project I've almost entirely moved on to Deep Learning and haven't written features like these anymore.

The dataset is not included but it is now available on the PAN website linked above. If you use this code or the paper,
you can cite the official paper:
```Bilan, Ivan, and Desislava Zhekova. "CAPS: A Cross-genre Author Profiling System." CLEF (Working Notes). 2016.```

# Installation Guide
**You will need:**

1) the Anaconda Python Package (Python 2.7) : https://www.continuum.io/downloads

2) gensim: https://radimrehurek.com/gensim/
to install, use these commands in anaconda:
conda install mingw libpython
conda install gensim

3) TreeTagger: http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/
First download the software and all needed language models from the website and then download a python wrapper
for TreeTagger: http://treetaggerwrapper.readthedocs.io/en/latest/

4) htmllaundry python module: 
pip install http://pypi.python.org/packages/source/h/htmllaundry/htmllaundry-2.0.tar.gz

5) textstat python module:
https://pypi.python.org/pypi/textstat/

The rest is included in the Anaconda package.


**How to use:**

To run the training software call the following command from command line (navigate to the current folder in cmd first):
$path_to_compiler train_cross_jenre.py -c $inputDataset -o $outputDir

To run the test software:
$path_to_compiler test_cross_jenre.py -c $inputDataset -r $inputRun -o $outputDir

$path_to_compiler variable denotes the full path to the Python 2.7 compiler (../python.exe).

$inputDataset variable denotes the full path to the input dataset. The format of the dataset must be similar to PAN14 or PAN16 datasets,
To run the software on the PAN15 dataset, use the software included in the "PAN15_compliant" Folder, since this dataset has a very different
structure compared to PAN14 and PAN16.

$outputDir in the training software is the path to were the models will be stored.

$inputRun in the test software is the path to the models (previously saved in the $outputDir).

$outputDir in the test software is the path to were the xml files with gender and age of the 
corresponding author will be saved.

The software recognizes the language automatically based on the appropriate language tag (en, es, nl) in the xml files of the current dataset in use.

The datasets are available at http://pan.webis.de/clef16/pan16-web/author-profiling.html


**How to evaluate:**

To evaluate the output of the model, use evaluator.py in the following way:
$path_to_compiler evaluator.py -c $inputFolder 

The evaluator outputs the results in form of accuracy, confusion matrix, precision, recall and F1-Score.
The labels are mapped into a numeric representation in the following way:

Gender: 
 "Gender Male : 1"
 "Gender Female : 2"
 
 Age: 
 "Age 18-24 : 3"
 "Age 25-34 : 4"
 "Age 35-49 : 5"
 "Age 50-64 : 6"
 "Age 65-xx : 7"
