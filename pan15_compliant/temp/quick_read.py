
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

def get_a_genre_mod_dev():
        main_dictionary = dict()
        number_of_files = 0
        tmp_list = list()
        author_id = 33
        count_same_text_file = 0

        filename = "C:/test_small/mini_dutch/12103872.xml"
        try:
            xmldoc = minidom.parse(filename)
            itemlist = xmldoc.getElementsByTagName('document')

            if len(itemlist) > 0:

                documents_per_author = 0
                bool_tester = 0

                for s in itemlist:

                    # print s.childNodes

                    number_of_files += 1
                    documents_per_author += 1

                    try:
                        # get CDATA
                        for node in s.childNodes:
                            # print node.nodeType
                            if node.nodeType == 4 or node.nodeType == 3:
                                # print "Getting the CDATA element of each author document"

                                text_inner = node.data.strip()
                                print text_inner

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
                                        current_document_sample = inner_soup.get_text()
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
                                        main_dictionary[author_id]['documents'] = [inner_soup.get_text()]

                                    bool_tester = 1
                                else:
                                    print "Error! The text sample is empty or couldn't read CDATA. Skipping to next one."

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
        # print "Found duplicates: ", count_same_text_file

        print main_dictionary
        print len(main_dictionary[33]['documents'])

get_a_genre_mod_dev()

