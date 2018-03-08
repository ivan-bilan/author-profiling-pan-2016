
__version__ = "1.0"
__date__ = "09.05.2016"
__author__ = "Ivan Bilan"

# -*- coding: utf-8 -*-

import glob
import ntpath
import re
import codecs
from nltk.tokenize import word_tokenize, sent_tokenize
from unidecode import unidecode
import pickle
import cPickle
from time import sleep
from xml.dom import minidom
from nltk.stem.porter import *
import treetaggerwrapper
from bs4 import BeautifulSoup
from itertools import tee
import HTMLParser

# pip install http://pypi.python.org/packages/source/h/htmllaundry/htmllaundry-2.0.tar.gz
from htmllaundry import strip_markup
from pylab import *

'''
from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordNeuralDependencyParser
path_to_jar = 'E:\\git\\stanford-corenlp-full-2015-12-09\\stanford-corenlp-3.6.0.jar'
path_to_models_jar = 'E:\\git\\stanford-corenlp-full-2015-12-09\\stanford-english-corenlp-2016-01-10-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
# dependency_parser = StanfordNeuralDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
'''

class PreprocessingClass(object):

    def __init__(self):

        # self.tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR="E:\\OtherSoftware\\TreeTagger")
        self.stemmer = PorterStemmer()
        self.h = HTMLParser.HTMLParser()
        # pass


    def get_a_genre_mod(self, path_to_folder, path_to_truth, main_dictionary, genre="Default Genre", lang=None):

        # print len(main_dictionary)

        # store truth here
        current_truth = dict()

        print "Reading ", genre

        # open truth file
        fr = codecs.open(path_to_truth, 'r', encoding="utf_8")

        for line in fr:

            current_line = line.strip().lstrip()
            if current_line:
                # print current_line
                current_id, current_gender, current_age = current_line.split(":::")[:3]
                if (current_gender == "M") or (current_gender == u"M"):
                    current_gender = 1
                elif (current_gender == "F") or (current_gender == u"F"):
                    current_gender = 2
                else:
                    current_gender = None

                if lang == "en" or lang == "es":
                    if (current_age == u"18-24") or (current_age == "18-24"):
                        current_age = 3
                    elif  (current_age == u"25-34") or (current_age == "25-34"):
                        current_age = 4
                    elif (current_age == u"35-49") or (current_age == "35-49"):
                        current_age = 5
                    elif (current_age == u"50-XX") or (current_age == "50-XX") or (current_age == u"50-xx") or (current_age == "50-xx") :
                        current_age = 6
                    else:
                        current_age = None

                    if (current_gender != None) and (current_age != None):

                        current_truth[current_id] = [current_gender, current_age]
                    else:
                        print
                        print "Error! Couldn't fully read the truth.txt."
                        print "Error at Author: ", current_truth, current_gender, current_age
                        # print current_line


                elif lang == "nl":
                    if current_gender != None:
                        current_truth[current_id] = [current_gender]
                    else:
                        print
                        print "Error! Couldn't fully read the truth.txt."
                        print "Error at Author: ", current_truth, current_gender
                        # print current_line
        fr.close()

        # finished reading truth

        # get files
        number_of_files = 0
        tmp_list = list()
        extracted_documents_per_author = 0

        count_same_text_file = 0

        for filename in path_to_folder:

            # print "Filename: ", filename
            author_id = ntpath.basename(filename).split(".")[0]
            # print author_id

            extracted_documents_per_author+=1

            try:
                xmldoc = minidom.parse(filename)
                itemlist = xmldoc.getElementsByTagName('document')

                if len(itemlist) > 0:

                    documents_per_author = 0
                    bool_tester = 0

                    for s in itemlist:

                        number_of_files += 1
                        documents_per_author += 1

                        try:
                            # get CDATA
                            for node in s.childNodes:
                                if node.nodeType == 4 or node.nodeType == 3:
                                    text_inner = node.data.strip()
                                    try:
                                        inner_soup = BeautifulSoup(text_inner, "lxml")
                                        # print inner_soup.get_text()
                                        # print
                                    except Exception as e:
                                        print filename
                                        print e

                                    if (len(inner_soup.get_text()) > 0) and (author_id in current_truth):
                                        if author_id in main_dictionary:
                                            pass
                                        else:
                                            main_dictionary[author_id] = dict()

                                        if 'documents' in main_dictionary[author_id]:
                                            current_document_sample = self.clean_my_file(inner_soup.get_text())
                                            if current_document_sample in main_dictionary[author_id]['documents']:
                                                count_same_text_file+=1

                                                '''
                                                print
                                                print author_id
                                                print main_dictionary[author_id]['documents']
                                                print current_document_sample
                                                sleep(100)
                                                '''
                                                pass
                                            else:
                                                main_dictionary[author_id]['documents'].append(current_document_sample)
                                        else:
                                            main_dictionary[author_id]['documents'] = [self.clean_my_file(inner_soup.get_text())]

                                        bool_tester = 1

                        except Exception as e:
                            print "Error! Failed to read a file."
                            print filename
                            print e
                            pass

                    tmp_list.append(documents_per_author)

                    if lang == "en" or lang == "es":

                        if (author_id in current_truth) and (bool_tester == 1):
                            main_dictionary[author_id]['gender'] = current_truth[author_id][0]
                            main_dictionary[author_id]['age'] = current_truth[author_id][1]
                            bool_tester = 0
                    elif lang == "nl":
                        if (author_id in current_truth) and (bool_tester == 1):
                            main_dictionary[author_id]['gender'] = current_truth[author_id][0]
                            bool_tester = 0

            except Exception as e:
                print
                print "Error! Couldn't read current text sample. Skipping to the next one."
                print "Error message: ", e
                print "Error occured in file: ", filename
                print
                pass

        try:
            average_blogs_per_author = float(sum(tmp_list))/len(tmp_list) if len(tmp_list) > 0 else float('nan')
        except:
            average_blogs_per_author = 'nan'

        # print len(current_truth), number_of_files, average_blogs_per_author
        # print len(main_dictionary)

        # for key, value in main_dictionary.iteritems():
            # print key, value
        print "Found duplicates: ", count_same_text_file
        return main_dictionary, len(main_dictionary), number_of_files, average_blogs_per_author


    def get_a_genre_mod_dev(self, path_to_folder, main_dictionary, genre="Default Genre", limit=None, limit_files=None):

        # print len(main_dictionary)

        # store truth here
        current_truth = dict()
        print "Reading ", genre

        # get files
        number_of_files = 0
        tmp_list = list()
        extracted_documents_per_author = 0

        count_same_text_file = 0

        for filename in path_to_folder:

            print "Filename: ", filename
            author_id = ntpath.basename(filename).split(".")[0]
            print "Extracted author id: ", author_id
            print "Data type of Author ID: ", type(author_id)
            try:
                author_id = str(author_id)
            except:
                pass
            print "Forced to String. Data type: ", author_id

            extracted_documents_per_author+=1

            try:
                xmldoc = minidom.parse(filename)
                itemlist = xmldoc.getElementsByTagName('document')

                if len(itemlist) > 0:

                    documents_per_author = 0
                    bool_tester = 0

                    for s in itemlist:

                        number_of_files += 1
                        documents_per_author += 1

                        try:
                            # get CDATA
                            for node in s.childNodes:
                                if node.nodeType == 4 or node.nodeType == 3:
                                    # print "Getting the CDATA element of each author document"
                                    text_inner = node.data.strip()
                                    try:
                                        inner_soup = BeautifulSoup(text_inner, "lxml")
                                        # print inner_soup.get_text()
                                        # print
                                    except Exception as e:
                                        print filename
                                        print e

                                    if (len(inner_soup.get_text()) > 0):
                                        if author_id in main_dictionary:
                                            pass
                                        else:
                                            main_dictionary[author_id] = dict()

                                        if 'documents' in main_dictionary[author_id]:
                                            current_document_sample = self.clean_my_file(inner_soup.get_text())
                                            if current_document_sample in main_dictionary[author_id]['documents']:
                                                count_same_text_file+=1

                                                '''
                                                print
                                                print author_id
                                                print main_dictionary[author_id]['documents']
                                                print current_document_sample
                                                sleep(100)
                                                '''
                                                pass
                                            else:
                                                main_dictionary[author_id]['documents'].append(current_document_sample)
                                        else:
                                            main_dictionary[author_id]['documents'] = [self.clean_my_file(inner_soup.get_text())]
                                        bool_tester = 1
                                    else:
                                        print "Error! The text sample is empty. Skipping to the next sample."
                                else:
                                    print "Error! The text sample is empty or couldn't read CDATA or plain text. Skipping to the next sample."

                        except Exception as e:
                            print "Error! Failed to read a file."
                            print filename
                            print e
                            pass

                    tmp_list.append(documents_per_author)

            except Exception as e:
                print
                print "Error! Couldn't read current text sample. Skipping to the next one."
                print "Error message: ", e
                print "Error occured in file: ", filename
                print
                pass

        try:
            average_blogs_per_author = float(sum(tmp_list))/len(tmp_list) if len(tmp_list) > 0 else float('nan')
        except:
            average_blogs_per_author = 'nan'

        # print len(current_truth), number_of_files, average_blogs_per_author
        # print len(main_dictionary)

        # for key, value in main_dictionary.iteritems():
            # print key, value
        print "Found duplicates: ", count_same_text_file

        return main_dictionary, len(main_dictionary), number_of_files, average_blogs_per_author


    def read_all_files(self, input_folder, set_comment, lang):

        # read files from the input folder
        folder_input_glob = glob.iglob(input_folder + """\\*.xml""")

        # print folder_input_glob

        if set_comment == "Training Set":
            read_input_truth = str(input_folder) + """\\truth.txt"""
            # print read_input_truth
            author_storage = dict()
            author_storage, counter_authors, number_of_samples, average_samples_per_author = self.get_a_genre_mod(folder_input_glob, read_input_truth, author_storage, set_comment, lang)


        elif set_comment == "Test Set":
            author_storage = dict()

            author_storage, counter_authors, number_of_samples, average_samples_per_author = self.get_a_genre_mod_dev(folder_input_glob, author_storage, set_comment)

        print "Dataset Statistics"
        print "Read Authors: ", counter_authors
        print "Read Text Samples: ", number_of_samples
        print "Text Samples per Author: ", average_samples_per_author
        # print len(author_storage)

        return author_storage


    def clean_my_file(self, x):

        #  preprocess the text
        # print x

        # get rid of newlines, tabs and carriage returns.
        x = re.sub('\r', '', x)
        x = re.sub('\t', '', x)
        x = re.sub('\n', '', x)

        # some of the blog posts have various html code elements in it's undecoded form,
        # some don't, we want to make sure that we get rid of all html code. That is why
        # we decode the most common html characters.

        # replace all linked content with [URL]
        # we will use the linked content in one of our features.
        x = re.sub('<[aA] (href|HREF)=.*?</[aA]>;?',' URL ', x) # replace urls
        x = re.sub('<img.*?>;?',' URL ', x) # replace urls
        x = re.sub('(http|https|ftp)://?[0-9a-zA-Z\.\/\-\_\?\:\=]*',' URL ',x)
        x = re.sub('(http|https|ftp)://?[0-9a-zA-Z\.\/\-\_\?\:\=]*',' URL ',x)
        x = re.sub('(^|\s)www\..+?(\s|$)', ' URL ', x)

        x = re.sub('(^|\s)(http|https|ftp)\:\/\/t\.co\/.+?(\s|$)', ' URL ', x)
        x = re.sub('(^|\s)(http|https|ftp)\:\/\/.+?(\s|$)', ' URL ', x)
        x = re.sub('(^|\s)pic.twitter.com/.+?(\s|$)', ' URL ', x)


        # clean all the HTML markups, this function is a part of htmllaundry
        x = strip_markup(x)

        # get rid of bbcode formatting and remaining html markups
        x = re.sub('[\[\<]\/?b[\]\>];?','', x)
        x = re.sub('[\[\<]\/?i[\]\>];?','', x)
        x = re.sub('[\[\<]br [\]\>];?','', x)
        x = re.sub('/>', '', x)
        x = re.sub('[\<\[]\/?h[1-4][\>\]]\;?','', x)
        x = re.sub('\[\/?img\]','', x)
        x = re.sub('\[\/?url\=?\]?','', x)
        x = re.sub('\[/?nickname\]','', x)
        # x = re.sub(';{1,}',' ', x)

        # get rid of whitespaces
        x = re.sub(' {1,}',' ', x)
        x = self.h.unescape(x)

        # delete everything else that strip_markup doesn't
        x = re.sub('height=".*?"','', x)
        x = re.sub('width=".*?"','', x)
        x = re.sub('alt=".*?"','', x)
        x = re.sub('title=".*?"','', x)
        x = re.sub('border=".*?"','', x)
        x = re.sub('align=".*?','', x)
        x = re.sub('style=".*?"','', x)
        x = re.sub(' otted  border-color:.*?"','', x)
        x = re.sub(' ashed  border-color:.*?"','', x)
        x = re.sub('target="_blank">','', x)
        x = re.sub('<a target=" _new"  href="  ]','', x)
        x = re.sub('<a target="_new" rel="nofollow" href=" ]','', x)

        # users for tweeter
        x = re.sub('(^|\s)@(?!\s).+?(?=(\s|$))', ' USER ', x)
        x = x.strip().lstrip()

        # print x
        return x


    def load_pickle_file(self, foldername, filename):
        # foldername = 'pickle_dataset/'
        file = foldername + filename
        loaded_file = open(file, 'rb')
        loaded_pickle_file = cPickle.load(loaded_file)
        loaded_file.close()
        return loaded_pickle_file


    def split_lists(self, author_dict, lang):

        if lang == "en" or lang == "es":
            X = list()
            y_gender = list()
            y_age = list()
            y_author = list()

            for key in author_dict:
                for all_documents in author_dict[key]['documents']:
                    X.append(all_documents)
                    y_gender.append(author_dict[key]['gender'])
                    y_age.append(author_dict[key]['age'])
                    y_author.append(key)

            return X, y_gender, y_age, y_author

        elif lang == "nl":
            X = list()
            y_gender = list()
            # y_age = list()
            y_author = list()

            for key in author_dict:
                for all_documents in author_dict[key]['documents']:
                    X.append(all_documents)
                    y_gender.append(author_dict[key]['gender'])
                    # y_age.append(author_dict[key]['age'])
                    y_author.append(key)

            return X, y_gender, y_author

    def split_lists_dev(self, author_dict):
        X = list()
        y_author = list()

        for key in author_dict:
            for all_documents in author_dict[key]['documents']:
                X.append(all_documents)
                y_author.append(key)

        return X, y_author


    def stem_and_pos(self, list_of_sentences, tagger, train_dev = 0):

        timer_start = datetime.datetime.now().replace(microsecond=0)
        print
        print "Started Tagging Text ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # read training
        tr_list_pos_tags = list()
        tr_list_lemma = list()
        tr_list_stems = list()

        for index, element in enumerate(list_of_sentences):
            # print
            # print "NEW SENTENCE"
            # print element
            # print tagger.tag_text(element)

            # token_element = word_tokenize(element)

            inner_tags = list()
            inner_lemmas = list()
            # inner_stems = list()

            # do stemming
            '''
            for word in token_element:
                try:
                    # print word, type(word)
                    stemmed_word = self.stemmer.stem(word)
                    # print stemmed_word
                    inner_stems.append(stemmed_word)

                except Exception as e:
                    # print e
                    pass
            '''

            # do pos tagging and lemmatization
            for s in tagger.tag_text(element):
                try:
                    inner_tags.append(s.split("\t")[1])
                except:
                    pass
                try:
                    inner_lemmas.append(s.split("\t")[2])
                except:
                    pass

            tr_list_pos_tags.append(" ".join(inner_tags))
            tr_list_lemma.append(" ".join(inner_lemmas))
            # tr_list_stems.append(" ".join(inner_stems))

        '''
        print "pos training"
        print len(list_of_sentences)
        print len(tr_list_pos_tags)
        print len(tr_list_lemma)
        '''

        # print len(tr_list_stems)
        timer_end = datetime.datetime.now().replace(microsecond=0)
        print "Finished Tagging Text ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print "Elapsed Time for Tagging: ", str(timer_end - timer_start)
        print

        return tr_list_pos_tags, tr_list_lemma



        '''
        if train_dev == 1:
            self.dump_feature_pickle(1, 'tr_sentences', list_of_sentences)
            self.dump_feature_pickle(1, 'tr_pos_tag', tr_list_pos_tags)
            self.dump_feature_pickle(1, 'tr_lemmas', tr_list_lemma)
            self.dump_feature_pickle(1, 'tr_stems', tr_list_stems)
        elif train_dev == 2:
            self.dump_feature_pickle(2, 'dev_sentences', list_of_sentences)
            self.dump_feature_pickle(2, 'dev_pos_tag', tr_list_pos_tags)
            self.dump_feature_pickle(2, 'dev_lemmas', tr_list_lemma)
            self.dump_feature_pickle(2, 'dev_stems', tr_list_stems)
        '''


    def create_pipeline_dict(self, sentence, stems, pos, author_list):
        list_of_dict = np.recarray(shape=(len(sentence),),
                                   dtype=[('author_id', object), ('raw_text', object), ('stem_text', object), ('pos_text', object)])

        for i, text in enumerate(sentence):
            list_of_dict['author_id'][i] = author_list[i]
            list_of_dict['raw_text'][i] = sentence[i]
            list_of_dict['stem_text'][i] = stems[i]
            list_of_dict['pos_text'][i] = pos[i]
        # print len(sentence), len(list_of_dict)
        return list_of_dict

    def create_pipeline_dict_single(self, choice_sample):

        print choice_sample
        print list(choice_sample)

        list_of_dict = np.recarray(shape=(1,),
                                   dtype=[('author_id', object), ('raw_text', object), ('stem_text', object), ('pos_text', object)])

        listed_choice = list(choice_sample)

        list_of_dict['author_id'][0] = listed_choice[0]
        list_of_dict['raw_text'][0] = listed_choice[1]
        list_of_dict['stem_text'][0] = listed_choice[2]
        list_of_dict['pos_text'][0] = listed_choice[3]

        return list_of_dict

    def get_dependancies(self, list_of_samples, train_dev, jenre):

        tweets_dependancy = list()
        tweets_dependancy_tree = list()

        final_errors = 0

        for index, sentence in enumerate(list_of_samples):
            if index % 500 == 0:
                print "current progress: ", index, " out of ", str(len(list_of_samples)) + " " + str(jenre)
                print "current errors: " + str(final_errors) + " " + jenre

            each_sentence = sent_tokenize(sentence)

            # print each_sentence

            if len(each_sentence) > 1:
                # print "more sentences"
                # print each_sentence, len(each_sentence)
                inner_tweets_dependancy = list()
                inner_tweets_dependancy_tree = list()

                for inner_sentence in each_sentence:
                    try:
                        split_words = inner_sentence.split()
                        if len(split_words) <= 100:

                            result, result2 = tee(dependency_parser.raw_parse(inner_sentence))
                            # print type(result)
                            # result2 = result
                            parse_tree = [parse.tree() for parse in result2]
                            # print parse_tree
                            # print

                            # print result
                            # print
                            dep = result.next()
                            # print dep
                            # print list(dep.triples())
                            '''
                            for element in list(dep.triples()):
                                print element
                            '''
                            final_parser = list(dep.triples())
                            # print parse_tree
                            # print
                        elif len(split_words) > 100:

                            result, result2 = tee(dependency_parser.raw_parse(" ".join(split_words[:100])))
                            # print type(result)
                            # result2 = result
                            parse_tree = [parse.tree() for parse in result2]
                            # print parse_tree
                            # print

                            # print result
                            # print
                            dep = result.next()
                            # print dep
                            # print list(dep.triples())
                            '''
                            for element in list(dep.triples()):
                                print element
                            '''
                            final_parser = list(dep.triples())
                            # print parse_tree
                            # print

                    except Exception as e:
                        print e
                        final_errors+=1
                        parse_tree = []
                        final_parser = []

                    # print "final errors: " + str(final_errors) + " " + jenre
                    inner_tweets_dependancy_tree.append(parse_tree)
                    inner_tweets_dependancy.append(final_parser)

                tweets_dependancy_tree.append(inner_tweets_dependancy_tree)
                tweets_dependancy.append(inner_tweets_dependancy)

                '''
                for inner_sentence in each_sentence:
                    # print inner_sentence
                    # result = dependency_parser.raw_parse(inner_sentence)
                    result, result2 = tee(dependency_parser.raw_parse(inner_sentence))
                    # result2 = dependency_parser.raw_parse(inner_sentence)
                    parse_tree = [parse.tree() for parse in result2]

                    dep = result.next()

                    # print list(dep.triples())
                    # print parse_tree
                    # print
                    inner_tweets_dependancy_tree.append(parse_tree)
                    inner_tweets_dependancy.append(list(dep.triples()))

                tweets_dependancy_tree.append(inner_tweets_dependancy_tree)
                tweets_dependancy.append(inner_tweets_dependancy)
                '''

            else:
                # print
                # print "one sentence"
                # result = dependency_parser.raw_parse(sentence)
                # result2 = dependency_parser.raw_parse(sentence)

                try:
                    split_words = sentence.split()
                    if len(split_words) <= 100:
                        result, result2 = tee(dependency_parser.raw_parse(sentence))
                        # print type(result)
                        # result2 = result
                        parse_tree = [parse.tree() for parse in result2]
                        # print parse_tree
                        # print

                        # print result
                        # print
                        dep = result.next()
                        # print dep
                        # print list(dep.triples())
                        '''
                        for element in list(dep.triples()):
                            print element
                        '''
                        final_parser = list(dep.triples())
                        # print parse_tree
                        # print
                    elif len(split_words) > 100:

                        result, result2 = tee(dependency_parser.raw_parse(" ".join(split_words[:100])))
                        # print type(result)
                        # result2 = result
                        parse_tree = [parse.tree() for parse in result2]
                        # print parse_tree
                        # print

                        # print result
                        # print
                        dep = result.next()
                        # print dep
                        # print list(dep.triples())
                        '''
                        for element in list(dep.triples()):
                            print element
                        '''
                        final_parser = list(dep.triples())
                        # print parse_tree
                        # print

                except Exception as e:
                    print e
                    final_errors+=1
                    parse_tree = []
                    final_parser = []


                tweets_dependancy.append(final_parser)
                tweets_dependancy_tree.append(parse_tree)

        print "final errors: " + str(final_errors) + " " + jenre

        print 'finished dependancies ' + jenre
        print "original: " + str(len(list_of_samples)) + " " + jenre
        print "parse: " + str(len(tweets_dependancy_tree)) + " " + jenre
        print "tree: " + str(len(tweets_dependancy)) + " " + jenre

        # sleep(10)
        print

        if jenre == "tweets":
            if train_dev == 1:
                self.dump_feature_pickle_tweets(1, 'tr_dependancy_parse', tweets_dependancy)
                self.dump_feature_pickle_tweets(1, 'tr_dependancy_trees', tweets_dependancy_tree)
            elif train_dev == 2:
                self.dump_feature_pickle_tweets(2, 'dev_dependancy_parse', tweets_dependancy)
                self.dump_feature_pickle_tweets(2, 'dev_dependancy_trees', tweets_dependancy_tree)
        elif jenre == "blogs":
            if train_dev == 1:
                self.dump_feature_pickle_blogs(1, 'tr_dependancy_parse', tweets_dependancy)
                self.dump_feature_pickle_blogs(1, 'tr_dependancy_trees', tweets_dependancy_tree)
            elif train_dev == 2:
                self.dump_feature_pickle_blogs(2, 'dev_dependancy_parse', tweets_dependancy)
                self.dump_feature_pickle_blogs(2, 'dev_dependancy_trees', tweets_dependancy_tree)
        elif jenre == "reviews":
            if train_dev == 1:
                self.dump_feature_pickle_reviews(1, 'tr_dependancy_parse', tweets_dependancy)
                self.dump_feature_pickle_reviews(1, 'tr_dependancy_trees', tweets_dependancy_tree)
            elif train_dev == 2:
                self.dump_feature_pickle_reviews(2, 'dev_dependancy_parse', tweets_dependancy)
                self.dump_feature_pickle_reviews(2, 'dev_dependancy_trees', tweets_dependancy_tree)
        # return tweets_dependancy, tweets_dependancy_tree


if  __name__ == '__main__':

    preprocessor = PreprocessingClass()

    # read all available files from all Datasets
    # preprocessor.read_all_files()

    # get a certain amount of samples from each dataset, preprocess them and store in a separate folder
    # preprocessor.get_sets()



    '''
    returned_new_dataset = preprocessor.read_final_dataset()

    training_sentences = returned_new_dataset[0]
    training_gender = returned_new_dataset[1]
    training_age = returned_new_dataset[2]
    training_author_id = returned_new_dataset[3]

    training_sentences_dev = returned_new_dataset[4]
    training_gender_dev = returned_new_dataset[5]
    training_age_dev = returned_new_dataset[6]
    training_author_id_dev = returned_new_dataset[7]
    '''

    '''
    for index, sentence in enumerate(training_sentences):
        if index % 100 == 0:
            print sentence
            print training_gender[index], training_age[index], training_author_id[index]

    for index, sentence in enumerate(training_sentences_dev):
        if index % 100 == 0:
            print sentence
            print training_gender_dev[index], training_age_dev[index], training_author_id_dev[index]
    '''

    '''
    preprocessor.stem_and_pos(training_sentences, 1)
    preprocessor.stem_and_pos(training_sentences_dev, 2)

    preprocessor.dump_feature_pickle(1, 'tr_gender_labels', training_gender)
    preprocessor.dump_feature_pickle(1, 'tr_age_labels', training_age)
    preprocessor.dump_feature_pickle(1, 'tr_author_id', training_author_id)

    preprocessor.dump_feature_pickle(2, 'dev_gender_labels', training_gender_dev)
    preprocessor.dump_feature_pickle(2, 'dev_age_labels', training_age_dev)
    preprocessor.dump_feature_pickle(2, 'dev_author_id', training_author_id_dev)
    '''