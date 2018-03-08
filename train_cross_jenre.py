# -*- coding: utf-8 -*-

__version__ = "1.0"
__date__ = "24.07.2016"
__author__ = "Ivan Bilan"

from nltk.stem.porter import *
from sklearn.linear_model import SGDClassifier
from sklearn import feature_selection
import treetaggerwrapper  # pip install treetaggerwrapper
from pylab import *
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
from PreprocessingClass import PreprocessingClass
from MainRunner import FinalClassicationClass

import xml.etree.ElementTree as ET
import os
import warnings
warnings.filterwarnings("ignore")


def detect_language(path, ):
    """
    Detect the language of the input dataset
    by getting the "lang" xml tag
    """
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
    """
    Create a command line options parser
    """
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-c", "--input", dest="input",
                      help="path/to/training/corpus")
    parser.add_option("-o", "--output", dest="output",
                      help="path/to/output/directory")
    return parser


def main(argv):
    parser = get_parser()

    # install treetagger and language models from http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/
    tagger_en = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR="C:\\TreeTagger")
    tagger_es = treetaggerwrapper.TreeTagger(TAGLANG='es', TAGDIR="C:\\TreeTagger")
    tagger_nl = treetaggerwrapper.TreeTagger(TAGLANG='nl', TAGDIR="C:\\TreeTagger")

    final_gender_classifier = LinearSVC(tol=1e-4,  C=0.10000000000000001, penalty='l2', class_weight={1: 1.0, 2: 0.9})

    '''
    # other classifiers for gender classification
    LinearSVC1 = SGDClassifier(n_iter=100, loss="hinge", penalty="l2")
    LinearSVC1 = RandomForestClassifier(n_estimators=300, criterion = 'entropy', max_features= 2000)
    LinearSVC1 = SVC(C = 0.10000000000000001, gamma = 0.0005, kernel='linear', probability=False, tol=1e-4, shrinking=True, cache_size=2000, class_weight={1:1.0, 2:0.9})
    '''

    final_age_classifier = OneVsRestClassifier(LogisticRegression(dual=False,multi_class='multinomial', solver='lbfgs'))
    preprocessor = PreprocessingClass()
    (options, args) = parser.parse_args(argv)

    if not (options.input and options.output):
        parser.error("Required arguments not provided")
    else:
        lang = detect_language(options.input)
        if lang.lower() not in ['en', 'es', 'nl']:
            print >> sys.stderr, 'Language other than en, es, nl'
            sys.exit(1)
        else:

            print "Current Language: ", lang

            try:
                new_output_path = re.sub("^/cygdrive/c/", "C:/", options.output)
            except Exception as e:
                print e
                new_output_path = options.output
            
            if lang.lower() == "en":
                main_classifier = FinalClassicationClass(lang.lower(), options.input, new_output_path, None, 1)
                dataset_input = preprocessor.read_all_files(options.input, "Training Set", lang.lower())
                X, y_gender, y_age, y_author = preprocessor.split_lists(dataset_input, lang.lower())
                # pos tagging and lemmatization
                X_list_pos_tags, X_list_lemma = preprocessor.stem_and_pos(X, tagger_en)
                # create a dictionary of text_samples, lemmatized_text_samples, pos_tagged_samples and author_ids
                pipelined_dictionary = preprocessor.create_pipeline_dict(X, X_list_lemma, X_list_pos_tags, y_author)
                main_classifier.dataset_statistics(X, y_gender, y_author, lang.lower(), y_age)
                pipeline_gender = main_classifier.make_pipeline_en(final_gender_classifier, "gender_tr", lang.lower())
                pipeline_age = main_classifier.make_pipeline_en(final_age_classifier, "age_tr", lang.lower())
                main_classifier.train_model(pipelined_dictionary, y_gender, y_author, pipeline_gender, pipeline_age, new_output_path, lang.lower(), y_age)
                # print options.input
                # print new_output_path
            elif lang.lower() == "nl":

                main_classifier = FinalClassicationClass(lang.lower(), options.input, new_output_path, None, 1)
                dataset_input = preprocessor.read_all_files(options.input, "Training Set", lang.lower())
                X, y_gender, y_author = preprocessor.split_lists(dataset_input, lang.lower())
                # pos tagging and lemmatization
                X_list_pos_tags, X_list_lemma = preprocessor.stem_and_pos(X, tagger_nl)
                # create a dictionary of text_samples, lemmatized_text_samples, pos_tagged_samples and author_ids
                pipelined_dictionary = preprocessor.create_pipeline_dict(X, X_list_lemma, X_list_pos_tags, y_author)
                main_classifier.dataset_statistics(X, y_gender, y_author, lang.lower())
                pipeline_gender = main_classifier.make_pipeline_nl(final_gender_classifier, "gender_tr", lang.lower())
                pipeline_age = main_classifier.make_pipeline_nl(final_age_classifier, "age_tr", lang.lower())
                main_classifier.train_model(pipelined_dictionary, y_gender, y_author, pipeline_gender, pipeline_age, new_output_path, lang.lower())
                # print options.input
                # print new_output_path
            elif lang.lower() == "es":
                main_classifier = FinalClassicationClass(lang.lower(), options.input, new_output_path, None, 1)
                dataset_input = preprocessor.read_all_files(options.input, "Training Set", lang.lower())
                X, y_gender, y_age, y_author = preprocessor.split_lists(dataset_input, lang.lower())
                # pos tagging and lemmatization
                X_list_pos_tags, X_list_lemma = preprocessor.stem_and_pos(X, tagger_es)
                # create a dictionary of text_samples, lemmatized_text_samples, pos_tagged_samples and author_ids
                pipelined_dictionary = preprocessor.create_pipeline_dict(X, X_list_lemma, X_list_pos_tags, y_author)
                main_classifier.dataset_statistics(X, y_gender, y_author, lang.lower(), y_age)
                pipeline_gender = main_classifier.make_pipeline_es(final_gender_classifier, "gender_tr", lang.lower())
                pipeline_age = main_classifier.make_pipeline_es(final_age_classifier, "age_tr", lang.lower())
                main_classifier.train_model(pipelined_dictionary, y_gender, y_author, pipeline_gender, pipeline_age, new_output_path, lang.lower(), y_age)

                # print options.input
                # print new_output_path


if __name__ == "__main__":
    main(sys.argv)
    sys.exit(1)
