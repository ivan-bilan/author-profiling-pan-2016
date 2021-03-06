# -*- coding: utf-8 -*-

__version__ = "alpha"
__date__ = "09.05.2016"
__author__ = "Ivan Bilan"


from itertools import tee
from sklearn.externals import joblib
# conda install mingw libpython
# conda install gensim
from gensim import corpora, models
from gensim.models import Phrases
from sklearn.externals import joblib
import pandas as pd
from pprint import pprint
from time import time
import logging
from time import gmtime, strftime
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.grid_search import GridSearchCV
from nltk.tokenize import word_tokenize
import pickle
import cPickle
import datetime
from nltk.stem.porter import *
# from stemming.porter2 import stem
# from sklearn.linear_model import SGDClassifier
from sklearn import feature_selection
# pip install treetaggerwrapper
import treetaggerwrapper
from random import random
# from multiprocessing import Process
import string, codecs
from time import sleep
import nltk
import re
import numpy
import scipy.sparse
from pylab import *
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# from multiprocessing import freeze_support

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import MultinomialNB
from textstat.textstat import textstat
# from AuthorProfiling.PAN_2016.working_files.backup_files.getMacroFScore_cross import getGenderFScore
import xml.etree.ElementTree as ET
from PreprocessingClass import PreprocessingClass
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer
from FeatureClass import FeatureClass
from sklearn.base import BaseEstimator, TransformerMixin
from MainRunner import FinalClassicationClass
from optparse import OptionParser
import os

import warnings
warnings.filterwarnings("ignore")
def detect_language(path, ):
    lang = ''
    for file in os.listdir(path):
        if file == 'truth.txt' or file == '.DS_Store':
            continue

        tree = ET.parse(os.path.join(path, file))
        root = tree.getroot()
        lang = root.get('lang')
        break
    return lang.strip()

def get_parser():
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-c", "--input", dest="input",
                      help="path/to/training/corpus")
    parser.add_option("-r", "--model", dest="model",
                      help="path/to/model/directory")
    parser.add_option("-o", "--output", dest="output",
                      help="path/to/output/directory")
    return parser


def load_model(input_folder, language):

    '''
    m = re.match("^C:/" , input_folder)
    if m:
        input_folder = re.sub("C:/", "/cygdrive/c/", input_folder, re.IGNORECASE)
        print "Changed InoutRun path to cygwin style: ", input_folder
    else:
        pass
    '''

    if language.lower() == "en" or language.lower() == "es":
        gender_filename = input_folder + "/" + language + "_gender_model.pkl"
        age_filename = input_folder + "/" + language + "_age_model.pkl"
        gender_model = joblib.load(gender_filename)
        age_model = joblib.load(age_filename)
        return gender_model, age_model

    elif language.lower() == "nl":
        gender_filename = input_folder + "/" + language.lower() + "_gender_model.pkl"
        gender_model = joblib.load(gender_filename)
        return gender_model


def change_path_to_windows_style(input):
    try:
        new_output_path = re.sub("^/cygdrive/c/", "C:/", input)
    except Exception as e:
        print e
        new_output_path = input

    return new_output_path

def main(argv):
    parser = get_parser()
    preprocessor = PreprocessingClass()
    tagger_en = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR="C:\\TreeTagger")
    tagger_es = treetaggerwrapper.TreeTagger(TAGLANG='es', TAGDIR="C:\\TreeTagger")
    tagger_nl = treetaggerwrapper.TreeTagger(TAGLANG='nl', TAGDIR="C:\\TreeTagger")

    (options, args) = parser.parse_args(argv)

    if not (options.input and options.output):
        parser.error("Required arguments not provided")
    else:
        lang = detect_language(options.input)
        if lang.lower() not in ['en', 'es', 'nl']:
            print >> sys.stderr, 'Language other than en, es, nl'
            sys.exit(1)
        else:
            print
            print "Current Language: ", lang

            # final_model_path = options.model

            final_model_path = change_path_to_windows_style(options.model)
            final_output_path = change_path_to_windows_style(options.output)

            if lang.lower() == "en":
                main_classifier = FinalClassicationClass(lang.lower(), options.input, final_output_path, None, 2)

                dataset_input = preprocessor.read_all_files(options.input, "Test Set", lang.lower())
                X, y_author = preprocessor.split_lists_dev(dataset_input)
                X_list_pos_tags, X_list_lemma = preprocessor.stem_and_pos(X, tagger_en)
                pipelined_dictionary = preprocessor.create_pipeline_dict(X, X_list_lemma, X_list_pos_tags, y_author)
                main_classifier.dataset_statistics_dev(X, y_author)
                # load models
                gender_model, age_model = load_model(final_model_path, lang.lower())
                main_classifier.test_model(gender_model, pipelined_dictionary, lang, y_author, final_output_path, age_model)

            elif lang.lower() == "nl":
                main_classifier = FinalClassicationClass(lang.lower(), options.input, final_output_path, None, 2)

                dataset_input = preprocessor.read_all_files(options.input, "Test Set", lang.lower())
                X, y_author = preprocessor.split_lists_dev(dataset_input)
                X_list_pos_tags, X_list_lemma = preprocessor.stem_and_pos(X, tagger_nl)
                pipelined_dictionary = preprocessor.create_pipeline_dict(X, X_list_lemma, X_list_pos_tags, y_author)
                main_classifier.dataset_statistics_dev(X, y_author)
                # load models
                gender_model = load_model(final_model_path, lang.lower())
                main_classifier.test_model(gender_model, pipelined_dictionary, lang, y_author, final_output_path)

            elif lang.lower() == "es":
                main_classifier = FinalClassicationClass(lang.lower(), options.input, final_output_path, None, 2)

                dataset_input = preprocessor.read_all_files(options.input, "Test Set", lang.lower())
                X, y_author = preprocessor.split_lists_dev(dataset_input)
                X_list_pos_tags, X_list_lemma = preprocessor.stem_and_pos(X, tagger_es)
                pipelined_dictionary = preprocessor.create_pipeline_dict(X, X_list_lemma, X_list_pos_tags, y_author)
                main_classifier.dataset_statistics_dev(X, y_author)
                # load models
                gender_model, age_model = load_model(final_model_path, lang.lower())
                main_classifier.test_model(gender_model, pipelined_dictionary, lang, y_author, final_output_path, age_model)

if  __name__ == '__main__':
    main(sys.argv)
    sys.exit(1)