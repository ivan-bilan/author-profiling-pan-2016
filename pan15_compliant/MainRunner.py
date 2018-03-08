# -*- coding: utf-8 -*-

__version__ = "alpha"
__date__ = "09.05.2016"
__author__ = "Ivan Bilan"

# from __future__ import print_function

# pip install -U nltk
from nltk.corpus import stopwords

from itertools import tee

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
# from random import random
import random
# from multiprocessing import Process
import string, codecs
from time import sleep
import nltk
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
from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import MultinomialNB
from textstat.textstat import textstat
# from AuthorProfiling.PAN_2016.working_files.backup_files.getMacroFScore_cross import getGenderFScore

from PreprocessingClass import PreprocessingClass
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer
from FeatureClass import FeatureClass
from sklearn.base import BaseEstimator, TransformerMixin


class SingleFeatureExtractor(object):

    def extract_single_feature_pipeline(self, sentence, feature_function, lowercase=None, remove_punc=None , feature_mode=None, feature_name=None, custom_ending=None, write_pickle=1):

        # print sentence
        # print len(sentence), type(sentence)
        remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

        list_single = list()
        # print

        if lowercase == 1:
            sentence = sentence.lower()
        elif lowercase == 2:
            pass

        if remove_punc == 1:
            sentence = sentence.translate(remove_punctuation_map)
        elif remove_punc == 2:
            pass

        # this is used for custom ending feature functions, 1 = skip, else = use specified ending string
        if custom_ending == 1:
            # print sentence
            if feature_mode == "single":
                return feature_function(sentence)
            elif feature_mode == "single_whole":
                return feature_function(sentence, 1)
            elif feature_mode == "single_pre_scaled":
                # print "working on suffix features"
                function_result = feature_function(sentence, 5)
                # print function_result
                return function_result
            # list_single.append([feature_function(sentence)])
        elif isinstance(custom_ending, basestring):
            if feature_mode == "single":
                return feature_function(sentence, 0, custom_ending)
            elif feature_mode == "single_whole":
                # print "working on suffix features"
                function_result = feature_function(sentence, 1, custom_ending)
                # print function_result
                return function_result
            elif feature_mode == "single_pre_scaled":
                # print "working on suffix features"
                function_result = feature_function(sentence, 5, custom_ending)
                # print function_result
                return function_result
            # list_single.append([feature_function(sentence, custom_ending)])
        elif custom_ending == 2:
            if feature_mode == "single_whole_ari":
                # print "single_whole_ari"
                try:
                    current_result = float(feature_function(sentence, 1))
                    if current_result < 0:
                        return 0
                    else:
                        return current_result
                except:
                    return 0
            else:
                try:
                    return feature_function(sentence)
                    # list_single.append([feature_function(sentence)])
                except:
                    return 0


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        # test output
        # print data_dict[self.key]
        # print data_dict[self.key][2]
        return data_dict[self.key]


class MeasureFeatures(BaseEstimator, SingleFeatureExtractor):
    """
    extends scikit learn BaseEstimator class
    """

    def __init__(self, comment, lang):
        self.comment = comment
        self.lang = lang

    def get_feature_names(self):
        return numpy.array(['type_token', 'avgwordlenght', 'punctuation', 'capitals'])


    def fit(self, documents, y=None):
        return self

    def transform(self, documents):

        unique_identifier = self.comment + "_" + str(len(documents))
        inner_feature_name = unique_identifier + "measure_cluster"
        print "Unique_identifier: ", unique_identifier

        X_average_word_len = list()
        X_type_token = list()
        X_punctuation = list()
        X_capitals = list()
        X_allcaps = list()

        print "Started training Measure features"
        timer_start = datetime.datetime.now().replace(microsecond=0)

        featureObject = FeatureClass(self.lang)
        for element in documents:
            # print element
            X_average_word_len.append(self.extract_single_feature_pipeline(element, featureObject.average_wordlength, 2, 2, "single", "average_word_len", 1))
            X_type_token.append(self.extract_single_feature_pipeline(element, featureObject.type_token_ratio, 1, 1, "single", "typetoken", 1))
            X_punctuation.append(self.extract_single_feature_pipeline(element, featureObject.general_punctuation_new, 2, 2, "single_pre_scaled", "punctuation", 1))
            X_capitals.append(self.extract_single_feature_pipeline(element, featureObject.words_capitalized, 2, 2, "single_pre_scaled", "capitals", 1))
            X_allcaps.append(self.extract_single_feature_pipeline(element, featureObject.AllCaps, 2, 2, "single_pre_scaled", "capitals", 1))

        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Training time for Measure features" + " :" + str(timer_end - timer_start)
        print

        X = numpy.array([X_type_token, X_average_word_len, X_punctuation, X_capitals, X_allcaps]).T

        # dump and load
        # dump_feature_for_all_sets(inner_feature_name, unique_identifier, X)
        # X = load_feature_for_all_sets(unique_identifier, inner_feature_name)

        print inner_feature_name, X
        print X.shape

        return X

def lda_preprocess(text):
    return text.lower()


class LDA(BaseEstimator, VectorizerMixin):
    """
    uses gensim library for topic modeling
    """
    def __init__(self, lang):

        self.number_of_topics = 100
        self.preprocessor = lda_preprocess
        self.tokenizer = None
        self.language = lang
        # print self.language

        if self.language == "en":
            self.stop_words = 'english'
        elif self.language == "nl":
            self.stop_words = 'dutch'
        elif self.language == "es":
            self.stop_words = 'spanish'

        self.bigram = True
        self.trigram = None
        self.analyzer = 'word'
        self.ngram_range = (1, 1)
        self.input = 'content'
        self.encoding = 'utf-8'
        self.strip_accents = None
        self.decode_error = 'strict'
        self.lowercase = True
        self.token_pattern = r"(?u)\b\w\w+\b"


    def get_stop_words(self):
        stop_lst = stopwords.words(self.stop_words)
        # print self.stop_words
        # stop_lst=[]
        # if self.stop == "english":
        # stop_lst = stopwords.words('english')

        # print stop_lst

        stop_lst.extend(["USER", "URL", "i'm", "rt"])
        stop_lst.extend(list(string.punctuation))
        # print "Stopword list------->",stop_lst
        return stop_lst

    def get_feature_names(self):
        return np.array(
            ["Topic_" + str(topic) for topic in xrange(0, self.number_of_topics)])


    def _build_vocabulary(self, raw_documents, fixed_vocab=None):

        analyze = self.build_analyzer()

        '''
        for document in raw_documents:
            print document
            print
        '''

        doc_token_lst = [analyze(document) for document in raw_documents]

        if self.bigram or self.trigram:
            bigram = Phrases(doc_token_lst)

        if self.trigram:
            trigram = []

        if fixed_vocab:
            vocabulary = self.vocabulary_
            tfidf_model = self.tfidf_
            corpus_vector = [vocabulary.doc2bow(text) for text in doc_token_lst]
        else:
            vocabulary = corpora.Dictionary(doc_token_lst)

            corpus_vector = [vocabulary.doc2bow(text) for text in doc_token_lst]

            tfidf_model = models.TfidfModel(corpus_vector)
            self.tfidf_ = tfidf_model

        return tfidf_model[corpus_vector], vocabulary

    def fit(self, documents, y=None):
        self.fit_transform(documents)
        return self


    def fit_transform(self, raw_documents, y=None):

        # analyze = self.build_analyzer()
        # doc_token_lst = [analyze(document) for document in raw_documents]  # bigrams of phrases
        # if self.bigram or self.trigram:
        # bigram = Phrases(doc_token_lst)
        #
        # if self.trigram:
        # trigram = []
        #
        # vocabulary = corpora.Dictionary(doc_token_lst)
        # self.corpus_vector_ = [self.dictionary.doc2bow(text) for text in doc_token_lst]
        #
        # self.tfidf_ = models.TfidfModel(self.corpus_vector_)
        # self.corpus_tfidf_ = self.tfidf_[self.corpus_vector_]


        X, vocabulary = self._build_vocabulary(raw_documents, False)
        self.vocabulary_ = vocabulary

        self.lda_ = models.LdaModel(X, id2word=self.vocabulary_, num_topics=self.number_of_topics)

        # self.lda_ = models.hdpmodel.HdpModel(X, id2word=self.vocabulary_) # , num_topics=self.number_of_topics


        return self._fit_doc_topic(X)


    def _fit_doc_topic(self, X):
        Topic_X = []
        for doc in X:
            weight = np.zeros(self.number_of_topics)
            for topic_id, prob in self.lda_[doc]:
                weight[topic_id] = prob
            Topic_X.append(weight)
        # print Topic_X

        return np.array(Topic_X)


    def topics(self, n_topic_words, out_file):
        with codecs.open(out_file, 'w') as out:
            for i in range(0, self.number_of_topics):
                topic_words = [term[1].encode('utf-8') for term in self.lda_.show_topic(i, n_topic_words)]
                out.write("Top {} terms for topic #{} : {}".format(n_topic_words, i, ", ".join(topic_words)))
                out.write("\n\n================================================================================\n")


    def transform(self, documents):
        # if not hasattr(self, 'vocabulary_'):
        # self._check_vocabulary()

        if not hasattr(self, 'vocabulary_') or len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary wasn't fitted or is empty!")

        X, _ = self._build_vocabulary(documents, True)

        return self._fit_doc_topic(X)



class FamilialFeatures(BaseEstimator, SingleFeatureExtractor):
    """
    extends scikit learn BaseEstimator class
    computes familial tokens features
    """

    def __init__(self, comment, lang):

        self.language = lang

        if self.language == "en":
            self.male = set([u'wife', u'gf', u'girlfriend'])
            self.female = set([u'husband', u'bf', u'boyfriend', u'hubby'])
            self.neutral = set([u'son', u'daughter', u'grandson', u'granddaughter', u'father', u'mother', u'brother', u'sister', u'uncle', u'aunt', u'cousin', u'nephew', u'niece', u'family', u'godson', u'goddaughter', u'grandchild', u'grandmother', u'grandfather', u'baby', u'babies', u'child', u'children', u'kids', u'kid', u'mom', u'parent'])
        elif self.language == "nl":
            self.male = set([u'vrouw', u'vriendin', u'lieve vrouw'])
            self.female = set([u'man', u'bf', u'vriend', u'lieve man', u'manlief', u'vriendje'])
            self.neutral = set([u'zoon', u'dochter', u'kleinzoon', u'kleindochter', u'vader', u'moeder', u'broer', u'zus', u'oom', u'tante', u'neef', u'nicht', u'niece', u'familie', u'petekind', u'kleinkind', u'grootmoeder', u'grootvader', u'baby', u'babys', u'kind', u'kinderen', u'ouder'])
        elif self.language == "es":
            self.male = set([u'esposa', u'novia', u'amiga', u'mujer', u'señora', u'vieja'])
            self.female = set([u'esposo', u'marido', u'esposito', u'novio', u'amigo', u'novio', u'maridito', u'hombre'])
            self.neutral = set([u'hijo', u'hija', u'hijos', u'hijas', u'nietos', u'nietas', u'papa', u'mama', u'abuelos', u'abuela', u'abuelo', u'hermano', u'hermana', u'tío', u'tía', u'primo', u'prima', u'sobrina', u'sobrino', u'crio', u'cría', u'bebes', u'familia', u'ahijado', u'ahijada', u'nieto', u'nieta', u'niños', u'niñas', u'mami', u'papi', u'pareja'])

        self.comment = comment
        self.preprocessing_counter = FeatureClass(self.language)


    def get_feature_names(self):
        return np.array(['male_bucket', 'female_bucket', 'neutral_bucket'])


    def fit(self, documents, y=None):
        return self

    def transform(self, documents):

        unique_identifier = self.comment + "_" + str(len(documents))
        inner_feature_name = unique_identifier + "familial_cluster"
        print "Unique_identifier: ", unique_identifier

        print "Started calculating Familial Features " + self.comment
        timer_start = datetime.datetime.now().replace(microsecond=0)
        featureObject = FeatureClass(self.language)
        # single
        male = re.compile(featureObject.regex_str(self.male), re.IGNORECASE)
        # male = re.compile(r"\b(?:dw|my\b\W+\b(?:girlfriend|gf|wife))\W*", re.IGNORECASE)
        # print regex_str(self.male)
        female = re.compile(featureObject.regex_str(self.female), re.IGNORECASE)
        neutral = re.compile(featureObject.regex_str(self.neutral), re.IGNORECASE)

        male_count = list()
        female_count = list()
        neutral_count = list()

        for index, doc in enumerate(documents):
            male_counter = len(male.findall(doc.lower()))
            female_counter = len(female.findall(doc.lower()))
            neutral_counter = len(neutral.findall(doc.lower()))

            male_count.append(self.preprocessing_counter.counter_pre_scaling(male_counter, len(doc.split())))
            female_count.append(self.preprocessing_counter.counter_pre_scaling(female_counter, len(doc.split())))
            neutral_count.append(self.preprocessing_counter.counter_pre_scaling(neutral_counter, len(doc.split())))

            '''
            from unidecode import unidecode
            print unidecode(doc)
            print male_count[index], female_count[index], neutral_count[index]
            '''

        # print neutral_count
        male_bucket = np.array(male_count)
        female_bucket = np.array(female_count)
        neutral_bucket = np.array(neutral_count)

        X = np.array([male_bucket, female_bucket, neutral_bucket]).T

        # dump and load
        # dump_feature_for_all_sets(inner_feature_name, unique_identifier, X)
        # X = load_feature_for_all_sets(unique_identifier, inner_feature_name)

        print X

        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Training time for Familial Cluster" + " :" + str(timer_end - timer_start) + " " + self.comment
        print

        return X



class CounterFeatures(BaseEstimator, SingleFeatureExtractor):

    def __init__(self, comment, lang):
        self.comment = comment
        self.language = lang

    def get_feature_names(self):
        return numpy.array(['connective_words', 'emotion_words', 'linked_content', 'stop_words', 'contractions', 'slang_words', 'abreviations'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):

        unique_identifier = self.comment + "_" + str(len(documents))
        inner_feature_name = unique_identifier + "counter_cluster"
        print "Unique_identifier: ", unique_identifier

        X_connectives = list()
        X_emotion_words =list()
        X_urls = list()
        X_stop_words = list()
        X_contractions = list()
        X_slang_words = list()
        X_abbreviations = list()

        print "Started training Counter Feature " + self.comment
        timer_start = datetime.datetime.now().replace(microsecond=0)
        featureObject = FeatureClass(self.language)

        for index, element in enumerate(documents):
            X_connectives.append(self.extract_single_feature_pipeline(element, featureObject.connective_words, 1, 1, "single_pre_scaled", "connectives", 1))
            X_emotion_words.append(self.extract_single_feature_pipeline(element, featureObject.emotion_words, 1, 1, "single_pre_scaled", "emotion_words", 1))
            X_urls.append(self.extract_single_feature_pipeline(element, featureObject.catch_url, 2, 2, "single_pre_scaled", "url", 1))
            X_stop_words.append(self.extract_single_feature_pipeline(element, featureObject.count_stop_words, 1, 2, "single_pre_scaled", "count_stop_words", 1))
            X_contractions.append(self.extract_single_feature_pipeline(element, featureObject.contractions, 1, 2, "single_pre_scaled", "contractions", 1))
            X_slang_words.append(self.extract_single_feature_pipeline(element, featureObject.slang_words, 2, 2, "single_pre_scaled", "slang_words", 1))
            X_abbreviations.append(self.extract_single_feature_pipeline(element, featureObject.get_abbreviations, 1, 1, "single_pre_scaled", "abbreviations", 1))

            '''
            from unidecode import unidecode
            print unidecode(element)
            print X_connectives[index], X_emotion_words[index], X_urls[index], X_stop_words[index], X_contractions[index], X_slang_words[index], X_abbreviations[index]
            '''

        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Training time for Counter Feature" + " :" + str(timer_end - timer_start)
        print

        X = numpy.array([X_connectives, X_emotion_words, X_urls, X_stop_words, X_contractions, X_slang_words, X_abbreviations]).T

        print inner_feature_name, X
        print X.shape
        # dump and load

        # dump_feature_for_all_sets(inner_feature_name, unique_identifier, X)
        # X = load_feature_for_all_sets(unique_identifier, inner_feature_name)

        # print "Counter Features: ", X
        return X


class CounterFeatures2(BaseEstimator, SingleFeatureExtractor):

    def __init__(self, comment, lang):
        self.comment = comment
        self.language = lang
        self.preprocessing_counter = FeatureClass(self.language)

    def get_feature_names(self):
        return numpy.array(['question_marks', 'users', 'hashes', 'exclam_mark'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):

        unique_identifier = self.comment + "_" + str(len(documents))
        inner_feature_name = unique_identifier + "counter_2_cluster"
        print "Unique_identifier: ", unique_identifier

        X_question_mark =list()
        X_exclamation_mark = list()
        X_hashes = list()
        X_users = list()

        print "Started training Counter Feature " + self.comment
        timer_start = datetime.datetime.now().replace(microsecond=0)

        for index, doc in enumerate(documents):

            question_counter = float(len(re.findall(r"\?", doc)))
            X_question_mark.append(self.preprocessing_counter.counter_pre_scaling_char(question_counter, len(doc)))

            exlam_mark_counter = float(len(re.findall(r"\!", doc)))
            X_exclamation_mark.append(self.preprocessing_counter.counter_pre_scaling_char(exlam_mark_counter, len(doc)))

            hash_counter = float(len(re.findall(r"\#", doc)))
            X_hashes.append(self.preprocessing_counter.counter_pre_scaling_char(hash_counter, len(doc)))

            users_counter = float(len(re.findall(r"user", doc, re.IGNORECASE)))
            X_users.append(self.preprocessing_counter.counter_pre_scaling(users_counter, len(doc.split())))

            '''
            from unidecode import unidecode
            print unidecode(doc)
            print X_question_mark[index], X_exclamation_mark[index], X_hashes[index], X_users[index]
            '''

        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Training time for Counter Feature" + " :" + str(timer_end - timer_start)
        print


        X = numpy.array([X_question_mark, X_exclamation_mark, X_hashes, X_users]).T

        print inner_feature_name, X
        print X.shape

        # dump and load
        # dump_feature_for_all_sets(inner_feature_name, unique_identifier, X)
        # X = load_feature_for_all_sets(unique_identifier, inner_feature_name)
        # print "Counter Features: ", X

        return X


class POSCluster(BaseEstimator, SingleFeatureExtractor):
    """
    extends scikit learn BaseEstimator class
    computes writing density and styles features
    """

    def __init__(self, comment, lang):

        self.comment = comment
        self.language = lang

    def get_feature_names(self):

        return numpy.array(["plurality", "lexical_f_measure", "determiner", "pronouns", "adjectives", "cardinals", "to_preposition", "conjunctions", "verbs"])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):


        unique_identifier = self.comment + "_" + str(len(documents))
        inner_feature_name = unique_identifier + "pos_cluster"
        print "Unique_identifier: ", unique_identifier


        X_plurality = list()
        X_lexical_f_measure = list()
        X_determiner = list()
        X_pronouns = list()
        X_adjectives = list()
        X_cardinals = list()
        X_to_preposition = list()
        X_conjunctions = list()
        X_verbs = list()

        print "Started training POS Cluster for: " + self.comment
        print "Length for: " + self.comment + ", is " + str(len(documents))

        timer_start = datetime.datetime.now().replace(microsecond=0)

        featureObject = FeatureClass(self.language)
        for index, element in enumerate(documents):

            # print element
            X_plurality.append(self.extract_single_feature_pipeline(element, featureObject.plurality, 2, 2, "single_pre_scaled", "plurality", 1))
            X_lexical_f_measure.append(self.extract_single_feature_pipeline(element, featureObject.lexical_Fmeasure_new, 2, 2, "single", "lexical_fmeasure", 1))
            X_determiner.append(self.extract_single_feature_pipeline(element, featureObject.each_part_of_speech, 2, 2, "single_pre_scaled", "determiner", "determiner"))
            X_pronouns.append(self.extract_single_feature_pipeline(element, featureObject.each_part_of_speech, 2, 2, "single_pre_scaled", "pronouns", "pronouns"))
            X_adjectives.append(self.extract_single_feature_pipeline(element, featureObject.each_part_of_speech, 2, 2, "single_pre_scaled", "adjectives", "adjectives"))
            X_cardinals.append(self.extract_single_feature_pipeline(element, featureObject.each_part_of_speech, 2, 2, "single_pre_scaled", "cardinal_num", "cardinal_num"))
            X_to_preposition.append(self.extract_single_feature_pipeline(element, featureObject.each_part_of_speech, 2, 2, "single_pre_scaled", "to_pos", "to_pos"))
            X_conjunctions.append(self.extract_single_feature_pipeline(element, featureObject.each_part_of_speech, 2, 2, "single_pre_scaled", "conjunctions", "conjunctions"))
            X_verbs.append(self.extract_single_feature_pipeline(element, featureObject.each_part_of_speech, 2, 2, "single_pre_scaled", "verbs", "verbs"))

            '''
            from unidecode import unidecode
            print unidecode(element)
            print X_plurality[index], X_lexical_f_measure[index], X_determiner[index], X_pronouns[index], X_adjectives[index], X_cardinals[index], X_to_preposition[index], X_conjunctions[index], X_verbs[index]
            '''

        X = numpy.array([X_plurality, X_lexical_f_measure, X_determiner, X_pronouns, X_adjectives, X_cardinals, X_to_preposition, X_conjunctions,  X_verbs]).T

        # dump and load
        # dump_feature_for_all_sets(inner_feature_name, unique_identifier, X)
        # X = load_feature_for_all_sets(unique_identifier, inner_feature_name)

        print inner_feature_name, X
        print X.shape
        # plotting
        plot_label = "POS Cluster"
        # plot_feature_for_all_sets(inner_feature_name, unique_identifier, X, plot_label)

        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Training time for POS Cluster" + " :" + str(timer_end - timer_start) + " " + self.comment
        print

        return X


class StylisticEndings(BaseEstimator, SingleFeatureExtractor):
    """
    extends scikit learn BaseEstimator class
    computes writing density and styles features
    """

    def __init__(self, comment, lang):
        self.comment = comment
        self.language = lang

    def get_feature_names(self):
        return numpy.array(['suf_able', 'suf_ful', 'suf_al', 'suf_ible', 'suf_ic', 'suf_ive', 'suf_less', 'suf_ous', 'suf_ly'])


    def fit(self, documents, y=None):
        return self

    def transform(self, documents):

        unique_identifier = self.comment + "_" + str(len(documents))
        inner_feature_name = unique_identifier + "stylistic_cluster"
        print "Unique_identifier: ", unique_identifier


        X_able = list()
        X_ful = list()
        X_al = list()
        X_ible = list()
        X_ic = list()
        X_ive = list()
        X_less = list()
        X_ous = list()
        X_ly = list()

        print "Started training Stylistic Suffix Features EN " + self.comment
        timer_start = datetime.datetime.now().replace(microsecond=0)

        # print len(documents)
        # print documents[2]
        # sleep(10)

        featureObject = FeatureClass(self.language)
        for index, element in enumerate(documents):


            result_able = self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "able", "able")
            X_able.append(result_able)
            X_ful.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ful", "ful"))
            X_al.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "al", "al"))
            X_ible.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ible", "ible"))
            X_ic.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ic", "ic"))
            X_ive.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ive", "ive"))
            X_less.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "less", "less"))
            X_ous.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ous", "ous"))
            X_ly.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ly", "ly"))


        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Training time for Stylistic Suffix Features EN " + " :" + str(timer_end - timer_start) + self.comment
        print

        X = numpy.array([X_able, X_ful, X_al, X_ible, X_ic, X_ive, X_less, X_ous, X_ly]).T


        # dump and load
        # dump_feature_for_all_sets(inner_feature_name, unique_identifier, X)
        # X = load_feature_for_all_sets(unique_identifier, inner_feature_name)

        print inner_feature_name, X
        print X.shape

        return X



class StylisticEndingsDutch(BaseEstimator, SingleFeatureExtractor):
    """
    extends scikit learn BaseEstimator class
    computes writing density and styles features
    """

    def __init__(self, comment, lang):
        self.comment = comment
        self.language = lang

    def get_feature_names(self):
        # https://en.wiktionary.org/wiki/Category:Dutch_suffixes
        return numpy.array(['jes', 'iek', 'eren'])
        # 'achtig', 'baar', 'haftig', 'isch', 'lijks', 'vol'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):

        unique_identifier = self.comment + "_" + str(len(documents))
        inner_feature_name = unique_identifier + "stylistic_cluster"
        print "Unique_identifier: ", unique_identifier


        X_able = list()
        X_ful = list()
        X_al = list()
        X_ible = list()
        X_ic = list()
        X_ive = list()
        X_less = list()
        X_ous = list()
        X_ly = list()

        print "Started training Stylistic Suffix Features NL " + self.comment
        timer_start = datetime.datetime.now().replace(microsecond=0)

        # print len(documents)
        # print documents[2]
        # sleep(10)

        featureObject = FeatureClass(self.language)
        for index, element in enumerate(documents):

            '''
            result_able = self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "achtig", "achtig")
            X_able.append(result_able)
            X_ful.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "baar", "baar"))
            X_al.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "haftig", "haftig"))
            X_ible.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "isch", "isch"))
            X_ic.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "lijks", "lijks"))
            X_ly.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "vol", "vol"))
            '''

            X_ive.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "jes", "jes"))
            X_less.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "iek", "iek"))
            X_ous.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "eren", "eren"))


            from unidecode import unidecode
            print unidecode(element)
            print X_less[index], X_ive[index], X_ous[index]

        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Training time for Stylistic Suffix Features NL " + " :" + str(timer_end - timer_start) + self.comment
        print
        # X_able, X_ful, X_al,  X_ly, X_ible, X_ic,
        X = numpy.array([X_ive, X_less, X_ous]).T


        # dump and load
        # dump_feature_for_all_sets(inner_feature_name, unique_identifier, X)
        # X = load_feature_for_all_sets(unique_identifier, inner_feature_name)

        print inner_feature_name, X
        print X.shape

        return X



class StylisticEndingsSpanish(BaseEstimator, SingleFeatureExtractor):
    """
    extends scikit learn BaseEstimator class
    computes writing density and styles features
    """

    def __init__(self, comment, lang):
        self.comment = comment
        self.language = lang

    def get_feature_names(self):
        # http://spanish.about.com/od/spanishvocabulary/a/intro_to_suffixes_2.htm
        return numpy.array(['ito', 'ada', 'anza', 'acho', 'acha', 'mente', 'ita', 'ote', 'dero'])


    def fit(self, documents, y=None):
        return self

    def transform(self, documents):

        unique_identifier = self.comment + "_" + str(len(documents))
        inner_feature_name = unique_identifier + "stylistic_cluster"
        print "Unique_identifier: ", unique_identifier


        X_able = list()
        X_ful = list()
        X_al = list()
        X_ible = list()
        X_ic = list()
        X_ive = list()
        X_less = list()
        X_ous = list()
        X_ly = list()

        print "Started training Stylistic Suffix Features ES" + self.comment
        timer_start = datetime.datetime.now().replace(microsecond=0)

        # print len(documents)
        # print documents[2]
        # sleep(10)

        featureObject = FeatureClass(self.language)
        for index, element in enumerate(documents):


            result_able = self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ito", "ito")
            X_able.append(result_able)
            X_ful.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ada", "ada"))
            X_al.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "anza", "anza"))
            X_ible.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "acho", "acho"))
            X_ic.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "acha", "acha"))
            X_ive.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "mente", "mente"))
            X_less.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ita", "ita"))
            X_ous.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "ote", "ote"))
            X_ly.append(self.extract_single_feature_pipeline(element, featureObject.stylistic_ending_custom, 1, 1, "single_pre_scaled", "dero", "dero"))


            from unidecode import unidecode
            print unidecode(element)
            print X_able[index], X_ful[index], X_al[index], X_ible[index], X_ic[index], X_ive[index], X_less[index], X_ous[index], X_ly[index]

        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Training time for Stylistic Suffix Features ES " + " :" + str(timer_end - timer_start) + self.comment
        print

        X = numpy.array([X_able, X_ful, X_al, X_ible, X_ic, X_ive, X_less, X_ous, X_ly]).T


        # dump and load
        # dump_feature_for_all_sets(inner_feature_name, unique_identifier, X)
        # X = load_feature_for_all_sets(unique_identifier, inner_feature_name)

        print inner_feature_name, X
        print X.shape

        return X


class CategoricalCharNgramsVectorizer(TfidfVectorizer):
    """
    extends scikit learn TfidfVectorizer class
    generates different categories of char n-grams and uses them as features

    6. Sapkota, U., Bethard, S., Montes, M., Solorio, T.: Not all character n-grams are created equal:A study in authorship attribution. In: Proceedings of the 2015 Conference of the North Amer-ican Chapter of the Association for Computational Linguistics: Human Language Technolo-gies. pp. 93–102. Association for Computational Linguistics, Denver, Colorado (May–June2015), http://www.aclweb.org/anthology/N15-1010

    """

    _slash_W = string.punctuation + " "

    _punctuation = r'''['\"“”-‘’.?!…,:;#\<\=\>@\(\)\*]'''
    _beg_punct = lambda self, x: re.match('^' + self._punctuation + '\w+', x)
    _mid_punct = lambda self, x: re.match(r'\w+' + self._punctuation + '(?:\w+|\s+)', x)
    _end_punct = lambda self, x: re.match(r'\w+' + self._punctuation + '$', x)

    # re.match is anchored at the beginning
    _whole_word = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (
        i == 0 or y[i - 1] in self._slash_W) and (i + n == len(y) or y[i + n] in self._slash_W)
    _mid_word = lambda self, x, y, i, n: not (
        re.findall(r'(?:\W|\s)', x) or i == 0 or y[i - 1] in self._slash_W or i + n == len(y) or y[
            i + n] in self._slash_W)
    _multi_word = lambda self, x: re.match('\w+\s\w+', x)

    _prefix = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (i == 0 or y[i - 1] in self._slash_W) and (
        not (i + n == len(y) or y[i + n] in self._slash_W))
    _suffix = lambda self, x, y, i, n: not (re.findall(r'(?:\W|\s)', x)) and (
        not (i == 0 or y[i - 1] in self._slash_W)) and (i + n == len(y) or y[i + n] in self._slash_W)
    _space_prefix = lambda self, x: re.match(r'''^\s\w+''', x)
    _space_suffix = lambda self, x: re.match(r'''\w+\s$''', x)

    # def _whole_word(self,x, y, i, n):
    # if i == 0 or y[i-1] in self._slash_W:
    # if i + n == len(y) or y[i+n] in self._slash_W:
    # print "here 2"
    # return True
    # return False
    #
    # def _mid_word(self,x, y, i, n):
    # if i == 0 or y[i-1] in self._slash_W or i + n == len(y) - 1 or y[i+n] in self._slash_W:
    # return False
    # return True

    def __init__(self, beg_punct=None, mid_punct=None, end_punct=None, whole_word=None, mid_word=None, multi_word=None,
                 prefix=None, suffix=None, space_prefix=None, space_suffix=None, all=None, **kwargs):

        # SPIELRAUM!!!
        super(CategoricalCharNgramsVectorizer, self).__init__(**kwargs)

        self.beg_punct = False
        self.mid_punct = False
        self.end_punct = False
        self.whole_word = False
        self.mid_word = True
        self.multi_word = True
        self.prefix = True
        self.suffix = False
        self.space_prefix = False
        self.space_suffix = False
        self.ngram_range = (3, 3)
        # self.max_features = 2000

    # def _get_word(self, text_document, i, n):
    # start, end = 0, len(text_document)
    # for j in xrange(i, -1, -1):
    # if text_document[j] == ' ' or text_document[j] in string.punctuation:
    # start = j + 1
    # break
    #
    #     for k in xrange(i + n, len(text_document)):
    #         if text_document[k] == ' ' or text_document[k] in string.punctuation:
    #             end = k
    #             break
    #
    #     return text_document[start: end]

    def _categorical_char_ngrams(self, text_document):
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        ngrams = []
        min_n, max_n = self.ngram_range
        # print min_n,max_n
        for n in xrange(min_n, min(max_n + 1, text_len + 1)):
            for i in xrange(text_len - n + 1):
                # check categories
                gram = text_document[i: i + n]
                added = False

                # punctuations
                if self.beg_punct and not added:
                    if self._beg_punct(gram):
                        ngrams.append(gram)
                        added = True

                if self.mid_punct and not added:
                    if self._mid_punct(gram):
                        ngrams.append(gram)
                        added = True

                if self.end_punct and not added:
                    if self._end_punct(gram):
                        ngrams.append(gram)
                        added = True

                # words

                if self.multi_word and not added:
                    if self._multi_word(gram):
                        ngrams.append(gram)
                        added = True

                if self.whole_word and not added:
                    if self._whole_word(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

                if self.mid_word and not added:
                    if self._mid_word(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

                # affixes
                if self.space_prefix and not added:
                    if self._space_prefix(gram):
                        ngrams.append(gram)
                        added = True

                if self.space_suffix and not added:
                    if self._space_suffix(gram):
                        ngrams.append(gram)
                        added = True

                if self.prefix and not added:
                    if self._prefix(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

                if self.suffix and not added:
                    if self._suffix(gram, text_document, i, n):
                        ngrams.append(gram)
                        added = True

        return ngrams

    def build_analyzer(self):
        preprocess = super(TfidfVectorizer, self).build_preprocessor()
        return lambda doc: self._categorical_char_ngrams(preprocess(self.decode(doc)))





class FinalClassicationClass(object):

    def __init__(self, language, input_folder, output_folder, model_input=None, train_dev_type=None):
        self.language = language
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_input = model_input
        self.train_dev_type = train_dev_type
        # self.scaler2 = StandardScaler()
        self.featureObject = FeatureClass(self.language)


    def fit_and_predict(self, train_labels, train_features, test_features, classifier):
        classifier.fit(train_features, train_labels)
        print "Starting to classify"
        return classifier.predict(test_features)


    def load_pickle_file(self, foldername, filename):
        # foldername = 'pickle_dataset/'
        file = foldername + filename
        loaded_file = open(file, 'rb')
        loaded_pickle_file = pickle.load(loaded_file)
        loaded_file.close()
        return loaded_pickle_file

    def load_cpickle(self, foldername, filename):
        # foldername = 'pickle_dataset/'
        file = foldername + filename
        loaded_file = open(file, 'rb')
        loaded_pickle_file = cPickle.load(loaded_file)
        loaded_file.close()
        return loaded_pickle_file


    def make_pipeline_en(self, classifier, comment="", lang=None):

        countVecWord = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, analyzer=u'word', sublinear_tf=True, use_idf = True, min_df=2, max_df=0.85, lowercase = True) # 4000, 0.85
        countVecWord_tags = TfidfVectorizer(ngram_range=(1, 4), max_features= 1000, analyzer=u'word', min_df=2, max_df=0.85, sublinear_tf=True, use_idf = True, lowercase = False) # 2000, 0.85

        lda = LDA(lang) # {'n_topics':8, 'stop_words':'english', 'lang': lang}
        categorial = CategoricalCharNgramsVectorizer()

        familial_features = FamilialFeatures(comment, lang)
        counter_features = CounterFeatures(comment, lang)
        counter_features_2 = CounterFeatures2(comment, lang)
        pos_cluster = POSCluster(comment, lang)
        stylistic_features = StylisticEndings(comment, lang)
        measure_features = MeasureFeatures(comment, lang)

        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=90)
        fs_vect = feature_selection.SelectPercentile(feature_selection.chi2, percentile=80)

        inner_scaler = StandardScaler()
        inner_scaler2 = StandardScaler()
        fs2 = feature_selection.SelectPercentile(feature_selection.f_classif)
        fs3 = feature_selection.VarianceThreshold()
        # fs4 = feature_selection.f_regression()
        # countVecWord_chars = TfidfVectorizer(ngram_range=(1, 4), max_features= 3000, analyzer=u'char', max_df=0.85, min_df=2)

        '''
        ('flesch_reading_ease_ari', Pipeline([
            ('selector', ItemSelector(key='raw_text')),
            ('flesch_reading_ease_ari_feature', flesch_reading_ease_features)
        ])),
        '''

        svc = classifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE

        # f5 = feature_selection.RFE(estimator=svc, n_features_to_select=2000, step=1) # good results
        f5 = feature_selection.RFE(estimator=svc, n_features_to_select=4000, step=1) # 2000

        '''
        ('categorial', Pipeline([
            ('selector', ItemSelector(key='stem_text')),
            ('categorial_inner', categorial),
            # ('percentile_feature_selection', fs)
        ])),
        '''

        # countVecWord_chars
        pipeline2 = Pipeline([
            ('union', FeatureUnion(
                    transformer_list=[

                    ('vectorized_pipeline', Pipeline([
                        ('union_vectorizer', FeatureUnion([

                            ('lda', Pipeline([
                                ('selector', ItemSelector(key='stem_text')),
                                ('lda_inner', lda),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('char', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('char_inner', categorial),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('stem_text', Pipeline([
                                ('selector', ItemSelector(key='stem_text')),
                                ('stem_tfidf', countVecWord),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('pos_text', Pipeline([
                                ('selector', ItemSelector(key='pos_text')),
                                ('pos_tfidf', countVecWord_tags),  # list of dicts -> feature matrix
                                # ('percentile_feature_selection', fs)
                            ])),

                        ])),
                            ('percentile_feature_selection', fs_vect),
                            # ('inner_scale', inner_scaler)
                            # ('inner_scale', inner_scaler),
                        ])),


                    ('custom_pipeline', Pipeline([
                        ('custom_features', FeatureUnion([

                            ('counter_features_2', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('counter_features_2_inner', counter_features_2)
                            ])),

                            ('measure_features', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('measure_features_inner', measure_features)
                            ])),

                            ('familial_features_1', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('familial_features', familial_features)
                            ])),

                            ('pos_cluster', Pipeline([
                                ('selector', ItemSelector(key='pos_text')),
                                ('pos_cluster_inner', pos_cluster)
                            ])),

                            # Pipeline for standard bag-of-words model for body
                            ('stylistic_features', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('stylistic_features_inner', stylistic_features)
                            ])),

                            ('counter_features', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('counter_features_inner', counter_features)
                            ])),

                        ])),
                            # ('percentile_feature_selection', fs),
                            ('inner_scale', inner_scaler),
                    ])),

                    ],

                    # weight components in FeatureUnion
                    # n_jobs=6,

                    transformer_weights={
                        'vectorized_pipeline': 0.8,  # 0.8,
                        'custom_pipeline': 1.0  # 1.0
                    },
            )),

            # ('percentile_feature_selection', fs3),
            # ('rfe_feature_selection', f5),

            ('clf', classifier),
            ])

        return pipeline2

    def make_pipeline_nl(self, classifier, comment="", lang=None):

        countVecWord = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, analyzer=u'word', sublinear_tf=True, use_idf = True, min_df=2, max_df=0.85, lowercase = True) # 4000, 0.85
        countVecWord_tags = TfidfVectorizer(ngram_range=(1, 4), max_features= 1000, analyzer=u'word', min_df=2, max_df=0.85, sublinear_tf=True, use_idf = True, lowercase = False) # 2000, 0.85

        lda = LDA(lang) # {'n_topics':8, 'stop_words':'english', 'lang': lang}
        categorial = CategoricalCharNgramsVectorizer()

        familial_features = FamilialFeatures(comment, lang)
        counter_features = CounterFeatures(comment, lang)
        counter_features_2 = CounterFeatures2(comment, lang)
        pos_cluster = POSCluster(comment, lang)
        stylistic_features = StylisticEndingsDutch(comment, lang)
        measure_features = MeasureFeatures(comment, lang)

        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=90)
        fs_vect = feature_selection.SelectPercentile(feature_selection.chi2, percentile=80)

        inner_scaler = StandardScaler()
        inner_scaler2 = StandardScaler()
        fs2 = feature_selection.SelectPercentile(feature_selection.f_classif)
        fs3 = feature_selection.VarianceThreshold()
        # fs4 = feature_selection.f_regression()
        # countVecWord_chars = TfidfVectorizer(ngram_range=(1, 4), max_features= 3000, analyzer=u'char', max_df=0.85, min_df=2)

        '''
        ('flesch_reading_ease_ari', Pipeline([
            ('selector', ItemSelector(key='raw_text')),
            ('flesch_reading_ease_ari_feature', flesch_reading_ease_features)
        ])),
        '''

        svc = classifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE

        # f5 = feature_selection.RFE(estimator=svc, n_features_to_select=2000, step=1) # good results
        f5 = feature_selection.RFE(estimator=svc, n_features_to_select=600, step=3) # 2000

        '''
        ('categorial', Pipeline([
            ('selector', ItemSelector(key='stem_text')),
            ('categorial_inner', categorial),
            # ('percentile_feature_selection', fs)
        ])),

        # Pipeline for standard bag-of-words model for body
        ('stylistic_features', Pipeline([
            ('selector', ItemSelector(key='raw_text')),
            ('stylistic_features_inner', stylistic_features)
        ])),

        '''

        # countVecWord_chars
        pipeline2 = Pipeline([
            ('union', FeatureUnion(
                    transformer_list=[

                    ('vectorized_pipeline', Pipeline([
                        ('union_vectorizer', FeatureUnion([

                            ('lda', Pipeline([
                                ('selector', ItemSelector(key='stem_text')),
                                ('lda_inner', lda),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('char', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('char_inner', categorial),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('stem_text', Pipeline([
                                ('selector', ItemSelector(key='stem_text')),
                                ('stem_tfidf', countVecWord),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('pos_text', Pipeline([
                                ('selector', ItemSelector(key='pos_text')),
                                ('pos_tfidf', countVecWord_tags),  # list of dicts -> feature matrix
                                # ('percentile_feature_selection', fs)
                            ])),

                        ])),
                            ('percentile_feature_selection', fs_vect),
                            # ('inner_scale', inner_scaler)
                            # ('inner_scale', inner_scaler),
                        ])),


                    ('custom_pipeline', Pipeline([
                        ('custom_features', FeatureUnion([

                            ('counter_features_2', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('counter_features_2_inner', counter_features_2)
                            ])),

                            ('measure_features', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('measure_features_inner', measure_features)
                            ])),

                            ('familial_features_1', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('familial_features', familial_features)
                            ])),

                            ('pos_cluster', Pipeline([
                                ('selector', ItemSelector(key='pos_text')),
                                ('pos_cluster_inner', pos_cluster)
                            ])),

                            ('counter_features', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('counter_features_inner', counter_features)
                            ])),

                        ])),
                            # ('percentile_feature_selection', fs),
                            ('inner_scale', inner_scaler),
                    ])),

                    ],

                    # weight components in FeatureUnion
                    # n_jobs=6,

                    transformer_weights={
                        'vectorized_pipeline': 0.8,  # 0.8,
                        'custom_pipeline': 1.0  # 1.0
                    },
            )),

            # ('percentile_feature_selection', fs3),
            # ('rfe_feature_selection', f5),
            ('clf', classifier),
            ])

        return pipeline2



    def make_pipeline_es(self, classifier, comment="", lang=None):

        countVecWord = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, analyzer=u'word', sublinear_tf=True, use_idf = True, min_df=2, max_df=0.85, lowercase = True) # 4000, 0.85
        countVecWord_tags = TfidfVectorizer(ngram_range=(1, 4), max_features= 1000, analyzer=u'word', min_df=2, max_df=0.85, sublinear_tf=True, use_idf = True, lowercase = False) # 2000, 0.85

        lda = LDA(lang) # {'n_topics':8, 'stop_words':'english', 'lang': lang}
        categorial = CategoricalCharNgramsVectorizer()

        familial_features = FamilialFeatures(comment, lang)
        counter_features = CounterFeatures(comment, lang)
        counter_features_2 = CounterFeatures2(comment, lang)
        pos_cluster = POSCluster(comment, lang)
        stylistic_features = StylisticEndingsSpanish(comment, lang)
        measure_features = MeasureFeatures(comment, lang)


        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=90)
        fs_vect = feature_selection.SelectPercentile(feature_selection.chi2, percentile=80)

        inner_scaler = StandardScaler()
        inner_scaler2 = StandardScaler()
        fs2 = feature_selection.SelectPercentile(feature_selection.f_classif)
        fs3 = feature_selection.VarianceThreshold()
        # fs4 = feature_selection.f_regression()
        # countVecWord_chars = TfidfVectorizer(ngram_range=(1, 4), max_features= 3000, analyzer=u'char', max_df=0.85, min_df=2)

        '''
        ('flesch_reading_ease_ari', Pipeline([
            ('selector', ItemSelector(key='raw_text')),
            ('flesch_reading_ease_ari_feature', flesch_reading_ease_features)
        ])),
        '''

        svc = classifier
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE

        # f5 = feature_selection.RFE(estimator=svc, n_features_to_select=2000, step=1) # good results
        f5 = feature_selection.RFE(estimator=svc, n_features_to_select=4000, step=1) # 2000

        '''
        ('categorial', Pipeline([
            ('selector', ItemSelector(key='stem_text')),
            ('categorial_inner', categorial),
            # ('percentile_feature_selection', fs)
        ])),

        # Pipeline for standard bag-of-words model for body
        ('stylistic_features', Pipeline([
            ('selector', ItemSelector(key='raw_text')),
            ('stylistic_features_inner', stylistic_features)
        ])),

        '''

        # countVecWord_chars
        pipeline2 = Pipeline([
            ('union', FeatureUnion(
                    transformer_list=[

                    ('vectorized_pipeline', Pipeline([
                        ('union_vectorizer', FeatureUnion([

                            ('lda', Pipeline([
                                ('selector', ItemSelector(key='stem_text')),
                                ('lda_inner', lda),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('char', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('char_inner', categorial),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('stem_text', Pipeline([
                                ('selector', ItemSelector(key='stem_text')),
                                ('stem_tfidf', countVecWord),
                                # ('percentile_feature_selection', fs)
                            ])),

                            ('pos_text', Pipeline([
                                ('selector', ItemSelector(key='pos_text')),
                                ('pos_tfidf', countVecWord_tags),  # list of dicts -> feature matrix
                                # ('percentile_feature_selection', fs)
                            ])),

                        ])),
                            ('percentile_feature_selection', fs_vect),
                            # ('inner_scale', inner_scaler)
                            # ('inner_scale', inner_scaler),
                        ])),


                    ('custom_pipeline', Pipeline([
                        ('custom_features', FeatureUnion([

                            ('counter_features_2', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('counter_features_2_inner', counter_features_2)
                            ])),

                            ('measure_features', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('measure_features_inner', measure_features)
                            ])),

                            ('familial_features_1', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('familial_features', familial_features)
                            ])),

                            ('pos_cluster', Pipeline([
                                ('selector', ItemSelector(key='pos_text')),
                                ('pos_cluster_inner', pos_cluster)
                            ])),

                            ('counter_features', Pipeline([
                                ('selector', ItemSelector(key='raw_text')),
                                ('counter_features_inner', counter_features)
                            ])),

                        ])),
                            # ('percentile_feature_selection', fs),
                            ('inner_scale', inner_scaler),
                    ])),

                    ],

                    # weight components in FeatureUnion
                    # n_jobs=6,

                    transformer_weights={
                        'vectorized_pipeline': 0.8,  # 0.8,
                        'custom_pipeline': 1.0  # 1.0
                    },
            )),

            # ('percentile_feature_selection', fs3),
            # ('rfe_feature_selection', f5),

            ('clf', classifier),
            ])

        return pipeline2

    def clf_custom_score(self, y_dev, y_pred, comment=""):

        full_path = "results\\results_rfe750.txt"

        print "The Accuracy (" + comment + ") (Linear SVC) is: ", metrics.accuracy_score(y_dev, y_pred), strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_dev, y_pred)
        print
        # print "Confusion Matrix (" + comment + ")"
        # print metrics.confusion_matrix(y_dev, y_pred)
        print

        print "Pandas Confusion Matrix(" + comment + ")", strftime("%Y-%m-%d %H:%M:%S", gmtime())
        import pandas as pd
        y_true = pd.Series(y_dev)
        y_pred = pd.Series(y_pred)

        print pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        print
        print "Classification report (" + comment + ")", strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print metrics.classification_report(y_dev, y_pred)
        print

        with codecs.open(full_path, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(str(strftime("%Y-%m-%d %H:%M:%S", gmtime())) + "\n")
            f.write("The Accuracy (" + comment + ") (Linear SVC) is: " + str(metrics.accuracy_score(y_dev, y_pred)) +"\n")
            f.write("Pandas Confusion Matrix(" + comment + ")" +"\n")
            f.write(str(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))+"\n")
            f.write("Classification report (" + comment + ")"+"\n")
            f.write(str(metrics.classification_report(y_dev, y_pred))+"\n")



    def create_pipeline_dict_single(self, choice_sample):
        list_of_dict = np.recarray(shape=(1,),
                                   dtype=[('author_id', object), ('raw_text', object), ('stem_text', object), ('pos_text', object)])

        # print choice_sample
        # print list(choice_sample)

        listed_choice = list(choice_sample)
        list_of_dict['author_id'][0] = listed_choice[0]
        list_of_dict['raw_text'][0] = listed_choice[1]
        list_of_dict['stem_text'][0] = listed_choice[2]
        list_of_dict['pos_text'][0] = listed_choice[3]

        return list_of_dict


    def get_author_classes_gender(self, dev_authors, dev_list_of_labels_gender, gender_model, X):

        dev_index_for_each_author = dict()
        for index, element in enumerate(dev_authors):

            if element in dev_index_for_each_author:
                dev_index_for_each_author[element].append(index)
            else:
                dev_index_for_each_author[element] = [index]

        # print dev_list_of_labels_gender

        # dev_index_for_each_author
        final_tr_dict = dict()
        for key, value in dev_index_for_each_author.iteritems():
            current_id_list = list()
            count_male = list()
            count_female = list()
            random_list_gender = [1,2]

            for index, element in enumerate(dev_list_of_labels_gender):
                # print value
                # print

                if index in value:
                    # print "Appending", index
                    current_id_list.append(dev_list_of_labels_gender[index])

                    if dev_list_of_labels_gender[index] == 1:
                        count_male.append(dev_list_of_labels_gender[index])
                    elif dev_list_of_labels_gender[index] == 2:
                        count_female.append(dev_list_of_labels_gender[index])

            # print count_male

            if len(count_male) > len(count_female):
                final_tr_dict[key] = 1
            elif len(count_male) < len(count_female):
                final_tr_dict[key] = 2
            else:
                # print dev_index_for_each_author[key], type(dev_index_for_each_author[key])
                # print list(dev_index_for_each_author[key])
                list_transformed = list(dev_index_for_each_author[key])
                import random
                # print X[4]
                # print random.choice(list_transformed)
                # print X[random.choice(list_transformed)]

                choice_sample = X[0]
                final_tr_dict[key] = gender_model.predict(self.create_pipeline_dict_single(choice_sample))[0]

            '''
            elif len(count_male) == len(count_female):
                # print "randint"
                random_gender = randint(1,2)
                final_tr_dict[key] = random_gender
            '''
        '''
        for key, value in final_tr_dict.iteritems():
            print key, value
        '''

        # return final_tr_dict

        '''
        newlist_dev = list()
        for key in sorted(final_tr_dict):
            newlist_dev.append(final_tr_dict[key])
        '''

        # print newlist_dev

        return final_tr_dict


    def get_author_classes_age(self, dev_authors, dev_list_of_labels_age, age_model, X):

        # print dev_list_of_labels_age
        dev_index_for_each_author = dict()
        for index, element in enumerate(dev_authors):
            if element in dev_index_for_each_author:
                dev_index_for_each_author[element].append(index)
            else:
                dev_index_for_each_author[element] = [index]

        # dev_index_for_each_author

        final_tr_dict = dict()
        for key, value in dev_index_for_each_author.iteritems():
            current_id_list = list()
            count_18 = list()
            count_25 = list()
            count_35 = list()
            count_50 = list()

            for index, element in enumerate(dev_list_of_labels_age):
                # print value
                # print

                if index in value:
                    # print "Appending", index
                    current_id_list.append(dev_list_of_labels_age[index])

                    if dev_list_of_labels_age[index] == 3:
                        count_18.append(dev_list_of_labels_age[index])
                    elif dev_list_of_labels_age[index] == 4:
                        count_25.append(dev_list_of_labels_age[index])
                    elif dev_list_of_labels_age[index] == 5:
                        count_35.append(dev_list_of_labels_age[index])
                    elif dev_list_of_labels_age[index] == 6:
                        count_50.append(dev_list_of_labels_age[index])
            # print count_male


            max_value = max(len(count_18), len(count_25), len(count_35), len(count_50))
            # print max_value, type(max_value)

            if max_value == len(count_18) and max_value != len(count_25) and max_value != len(count_35) and max_value != len(count_50) and max_value:
                final_tr_dict[key] = 3
            elif max_value != len(count_18) and max_value == len(count_25) and max_value != len(count_35) and max_value != len(count_50) and max_value:
                final_tr_dict[key] = 4
            elif max_value != len(count_18) and max_value != len(count_25) and max_value == len(count_35) and max_value != len(count_50) and max_value:
                final_tr_dict[key] = 5
            elif max_value != len(count_18) and max_value != len(count_25) and max_value != len(count_35) and max_value == len(count_50) and max_value:
                final_tr_dict[key] = 6
            else:
                # import random
                list_transformed = list(dev_index_for_each_author[key])
                # int(random.choice(list_transformed))
                choice_sample = X[0]
                final_tr_dict[key] = age_model.predict(self.create_pipeline_dict_single(choice_sample))[0]


        '''
        for key, value in final_tr_dict.iteritems():
            print key, value
        '''

        '''
        newlist_dev = list()
        for key in sorted(final_tr_dict):
            newlist_dev.append(final_tr_dict[key])
        '''

        # print final_tr_dict

        return final_tr_dict



    def train_model(self, X, y_gender, y_author, pipeline_gender, pipeline_age, output_folder, lang, y_age=None):

        if lang.lower() == "en" or lang.lower() == "es":
            pipeline_gender.fit(X, y_gender)

            import os
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            joblib_name_gender = output_folder + "/" + lang.lower() + "_gender_model.pkl"
            joblib.dump(pipeline_gender, joblib_name_gender)

            # newlist_dev_gender = get_author_classes_gender(dev_authors, dev_list_of_labels_gender)
            # newlist_pred_gender = get_author_classes_gender(dev_authors, y_pred_gender)

            # fit pipeline for age classification
            pipeline_age.fit(X, y_age)

            joblib_name_age = output_folder + "/" + lang.lower() + "_age_model.pkl"
            joblib.dump(pipeline_age, joblib_name_age)

        elif lang.lower() == "nl":
            pipeline_gender.fit(X, y_gender)

            # joblib_name_gender = output_folder + "/" + lang.lower() + "_gender_model.pkl"

            '''
            import os
            filename = str(lang.lower()) + "_gender_model.pkl"
            print filename
            joblib_name_gender = os.path.join(output_folder, filename)
            print joblib_name_gender
            joblib.dump(pipeline_gender, joblib_name_gender)
            '''

            print
            print
            print
            joblib_name_gender = output_folder + "/" + lang.lower() + "_gender_model.pkl"

            import os
            if not os.path.exists(output_folder):
                print "Folder doesn't exist. Creating new folder: ", output_folder
                os.makedirs(output_folder)

            else:
                print "Folder already exists."

            _ = joblib.dump(pipeline_gender, joblib_name_gender)

            import os.path
            if os.path.exists(joblib_name_gender):
                print "File successfully created: ", joblib_name_gender
                # print "Contents of ", output_folder
                # print os.listdir(output_folder)
            else:
                print "Error! No model file created!"
                # print os.listdir(output_folder)


            test_list = [12,4,5,6,7]
            test_pickle_file_name = output_folder + "/" + 'test_file.pkl'
            with open(test_pickle_file_name, 'wb') as f:
                import cPickle
                cPickle.dump(test_list, f)
                print f


            with open(test_pickle_file_name, 'rb') as f:
                list_file = cPickle.load(f)

            print
            print "Contents of the pickle file test_file.pkl: ", list_file

            test_string = "Test text string"

            test_text_file_name = output_folder + "/" + 'test_file2.txt'
            with open(test_text_file_name, 'wb') as f:
                import cPickle
                cPickle.dump(test_string, f)
                # print f

            with open(test_text_file_name, 'rb') as f:
                list_file = cPickle.load(f)

            print
            print "Contents of test_file2.txt: ", test_string

            # print "Contents of ", output_folder
            # print os.listdir(output_folder)


        print
        print "Contents of ", output_folder
        print os.listdir(output_folder)

        # y_pred_age = pipeline_age.predict(list_of_dict_dev)
        # clf_custom_score(dev_list_of_labels_age, y_pred_age, comment)


        '''
        tr_index_for_each_author = dict()
        for index, element in enumerate(tr_authors):
            if tr_index_for_each_author[element]:
                tr_index_for_each_author[element] = [index]
            else:
                tr_index_for_each_author[element].append(index)
        '''
        '''
        newlist_dev = list()
        for key in sorted(final_dev_dict_gender):
            newlist_dev.append(final_dev_dict_gender[key])

        newlist_pred = list()
        for key in sorted(final_pred_dict_gender):
            newlist_pred.append(final_pred_dict_gender[key])
        '''

        '''
        newlist_dev_age = get_author_classes_age_test(dev_authors, dev_list_of_labels_age)
        newlist_pred_age = get_author_classes_age_test(dev_authors, y_pred_age)

        # test keys
        newlist_dev_test = list()
        for key in sorted(newlist_dev_age):
            newlist_dev_test.append(key)

        newlist_pred_test = list()
        for key in sorted(newlist_pred_age):
            newlist_pred_test.append(key)

        print newlist_dev_test, newlist_pred_test
        '''

        '''
        newlist_dev_age = get_author_classes_age(dev_authors, dev_list_of_labels_age)
        newlist_pred_age = get_author_classes_age(dev_authors, y_pred_age)

        #print newlist_dev_age
        # print
        # print newlist_pred_age
        new_comment = comment + " age per author"
        clf_custom_score(newlist_dev_age, newlist_pred_age, new_comment)
        '''


    def generate_final_xml(self, output_folder, language, predicted_gender, predicted_age=None):
        import xml.etree.cElementTree as ET
        '''
          <author id="{author-id}"
              type="to be announced"
              lang="en|es|nl"
              age_group="to be announced"
              gender="male|female"
          />
        '''

        # print predicted_gender
        # print predicted_age

        if predicted_age is None:

            for author in predicted_gender:

                # current_author_id = "{"+ str(author)+"}"
                current_author_id = str(author)

                current_age_group = "no age"

                if predicted_gender[author] == 1:
                    current_gender = "male"
                elif predicted_gender[author] == 2:
                    current_gender = "female"

                current_type = "cross-jenre"
                current_language = language.lower()

                root = ET.Element("author", id=current_author_id, type=current_type, lang=current_language, age_group=current_age_group, gender=current_gender)

                tree = ET.ElementTree(root)
                tree.write(output_folder + "/" + str(author) + ".xml")
        else:
            for author in predicted_gender:
                # current_author_id = "{"+ str(author)+"}"
                current_author_id = str(author)
                # print predicted_age[author]

                if predicted_age[author] == 3:
                    current_age_group = "18-24"
                elif predicted_age[author] == 4:
                    current_age_group = "25-34"
                elif predicted_age[author] == 5:
                    current_age_group = "35-49"
                elif predicted_age[author] == 6:
                    current_age_group = "50-xx"


                if predicted_gender[author] == 1:
                    current_gender = "male"
                elif predicted_gender[author] == 2:
                    current_gender = "female"

                current_type = "cross-jenre"
                current_language = language.lower()

                root = ET.Element("author", id=current_author_id, type=current_type, lang=current_language, age_group=current_age_group, gender=current_gender)

                tree = ET.ElementTree(root)
                tree.write(output_folder + "/" + str(author) + ".xml")



    def test_model(self, gender_model, X, language, y_author, output_folder, age_model=None):

        if age_model is None and language.lower() == "nl":

            y_pred_gender = gender_model.predict(X)

            newlist_pred_gender = self.get_author_classes_gender(y_author, y_pred_gender, gender_model, X)

            # print newlist_pred_gender, type(newlist_pred_gender)
            # print newlist_pred_age, type(newlist_pred_age)

            self.generate_final_xml(output_folder, language, newlist_pred_gender)

        elif language.lower() == "es" or language.lower() == "en":
            y_pred_gender = gender_model.predict(X)
            y_pred_age = age_model.predict(X)

            newlist_pred_gender = self.get_author_classes_gender(y_author, y_pred_gender, gender_model, X)
            newlist_pred_age = self.get_author_classes_age(y_author, y_pred_age, age_model, X)

            # print newlist_pred_gender, type(newlist_pred_gender)
            # print newlist_pred_age, type(newlist_pred_age)

            self.generate_final_xml(output_folder, language, newlist_pred_gender, newlist_pred_age)



    def dependancy_multiprocess(self, list_of_samples, train_dev, jenre):
        preprocessor = PreprocessingClass()
        preprocessor.get_dependancies(list_of_samples, train_dev, jenre)

    def dataset_statistics_dev(self, X, y_author):

        print
        print "Final Dataset Statistics"
        print "Number of Text Samples: ", len(X)
        print "Number of Unique Authors: ", len(set(y_author))


    def dataset_statistics(self, X, y_gender, y_author, lang, y_age=None):

        print
        print "Final Dataset Statistics"
        print "Number of Text Samples: ", len(X)
        print "Number of Unique Authors: ", len(set(y_author))
        print "Number of Male Text Samples: ", y_gender.count(1)
        print "Number of Female Text Samples: ", y_gender.count(2)

        if y_age is None:
            pass
        else:
            print "Number of 18-24 Age Group Text Samples: ", y_age.count(3)
            print "Number of 25-34 Age Group Text Samples: ", y_age.count(4)
            print "Number of 35-49 Age Group Text Samples: ", y_age.count(5)
            print "Number of 50-xx Age Group Text Samples: ", y_age.count(6)
            # print "Number of 65-xx Age Group Text Samples: ", y_age.count(7)


        '''
        for index, element in enumerate(tr_list_of_sentences_reviews):
            if index % 2400 == 0:
                print element
                print tr_list_of_labels_gender_reviews[index]
                print tr_list_of_labels_age_reviews[index]
                print tr_list_of_author_id_reviews[index]
                print tr_list_of_pos_reviews[index]
                print tr_list_of_stems_reviews[index]
                print tr_list_of_lemmas_reviews[index]
                # sleep(1)
        # sleep(120)

        for index, element in enumerate(dev_list_of_sentences_reviews):
            if index % 1400 == 0:
                print element
                print dev_list_of_labels_gender_reviews[index]
                print dev_list_of_labels_age_reviews[index]
                print dev_list_of_author_id_reviews[index]
                print dev_list_of_pos_reviews[index]
                print dev_list_of_stems_reviews[index]
                print dev_list_of_lemmas_reviews[index]
                # sleep(1)
        '''


if __name__ == '__main__':
    print "idle..."




