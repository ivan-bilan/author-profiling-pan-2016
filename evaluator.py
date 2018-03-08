# -*- coding: utf-8 -*-

__author__ = "Ivan Bilan"

# This is a custom evaluator
# All results have been evaluated on the official submission system of the PAN shared task at http://www.tira.io
# To evaluate results truth.txt should exist in the folder and it should include all final author labels

import glob
from xml.dom import minidom
import codecs
import re
from pylab import *
import xml.etree.ElementTree as ET
from PreprocessingClass import PreprocessingClass
from MainRunner import FinalClassicationClass
import os
import warnings
warnings.filterwarnings("ignore")


def detect_language(path):
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
    return parser


def change_path_to_windows_style(input):
    try:
        new_output_path = re.sub("^/cygdrive/c/", "C:/", input)
    except Exception as e:
        print e
        new_output_path = input

    return new_output_path


def get_truth(input_folder, path_to_truth, lang):
    # open truth file

    current_truth = dict()
    read_input_truth = str(input_folder) + """\\truth.txt"""
    fr = codecs.open(read_input_truth, 'r', encoding="utf_8")

    for line in fr:

        current_line = line.strip().lstrip()
        if current_line:
            # print current_line
            current_id, current_gender, current_age = current_line.split(":::")
            if (current_gender == "MALE") or (current_gender == u"MALE"):
                current_gender = 1
            elif (current_gender == "FEMALE") or (current_gender == u"FEMALE"):
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
                elif (current_age == u"50-64") or (current_age == "50-64"):
                    current_age = 6
                elif (current_age == u"65-xx") or (current_age == u"65-XX") or (current_age == "65-xx") or (current_age == "65-XX"):
                    current_age = 7
                else:
                    current_age = None

                if (current_gender is not None) and (current_age is not None):
                    current_truth[current_id] = [current_gender, current_age]
                else:
                    print
                    print "Error! Couldn't fully read the truth.txt."
                    print "Error at Author: ", current_truth, current_gender, current_age
                    # print current_line

            elif lang == "nl":
                if current_gender is not None:
                    current_truth[current_id] = [current_gender]
                else:
                    print
                    print "Error! Couldn't fully read the truth.txt."
                    print "Error at Author: ", current_truth, current_gender
                    # print current_line
    fr.close()
    return current_truth


def get_results(input_folder, lang):

    folder_input_glob = glob.iglob(input_folder + """/*.xml""")

    result_dict = dict()

    for element in folder_input_glob:
        # print element

        if lang == "en" or lang == "es":

            xmldoc = minidom.parse(element)
            itemlist = xmldoc.getElementsByTagName('author')

            # print itemlist

            for item in itemlist:

                author_id = item.attributes['id'].value

                if item.attributes['gender'].value == "female":
                    final_gender = 2
                elif item.attributes['gender'].value == "male":
                    final_gender = 1

                if (item.attributes['age_group'].value == u"18-24") or (item.attributes['age_group'].value == "18-24"):
                    current_age = 3
                elif (item.attributes['age_group'].value == u"25-34") or (item.attributes['age_group'].value == "25-34"):
                    current_age = 4
                elif (item.attributes['age_group'].value == u"35-49") or (item.attributes['age_group'].value == "35-49"):
                    current_age = 5
                elif (item.attributes['age_group'].value == u"50-64") or (item.attributes['age_group'].value == "50-64"):
                    current_age = 6
                elif (item.attributes['age_group'].value == u"65-xx") or (item.attributes['age_group'].value == u"65-XX") or (item.attributes['age_group'].value == "65-xx") or (item.attributes['age_group'].value == "65-XX"):
                    current_age = 7

                if author_id in result_dict:
                    pass
                else:
                    # print author_id, final_gender, current_age
                    result_dict[author_id] = [final_gender, current_age]

        else:
            xmldoc = minidom.parse(element)
            itemlist = xmldoc.getElementsByTagName('author')
            for item in itemlist:

                author_id = item.attributes['id'].value

                if item.attributes['gender'].value == "female":
                    final_gender = 2
                elif item.attributes['gender'].value == "male":
                    final_gender = 1

                if author_id in result_dict:
                    pass
                else:
                    result_dict[author_id] = [final_gender]

    return result_dict


def main(argv):
    parser = get_parser()
    (options, args) = parser.parse_args(argv)

    if not options.input:
        parser.error("Required arguments not provided")
    else:
        lang = detect_language(options.input)
        if lang.lower() not in ['en', 'es', 'nl']:
            print >> sys.stderr, 'Language other than en, es, nl'
            sys.exit(1)
        else:

            print "Class labels code map explained: "
            print "Gender Male : 1"
            print "Gender Female : 2"
            print "Age 18-24 : 3"
            print "Age 25-34 : 4"
            print "Age 35-49 : 5"
            print "Age 50-64 : 6"
            print "Age 65-xx : 7"

            main_classifier = FinalClassicationClass(lang, options.input, options.input, None, 2)

            if lang == "en" or lang == "es":

                truth_file_tmp = get_truth(options.input, options.input, lang)
                result_dict = get_results(options.input, lang)

                truth_file = dict()
                for element in truth_file_tmp:
                    # print element
                    if element in result_dict:
                        truth_file[element] = truth_file_tmp[element]

                newlist_gold_gender = list()
                for key in sorted(truth_file):
                    newlist_gold_gender.append(truth_file[key][0])

                newlist_gold_age = list()
                for key in sorted(truth_file):
                    newlist_gold_age.append(truth_file[key][1])

                newlist_pred_gender = list()
                for key in sorted(truth_file):
                    newlist_pred_gender.append(result_dict[key][0])

                newlist_pred_age = list()
                for key in sorted(truth_file):
                    newlist_pred_age.append(result_dict[key][1])

                new_comment = "gender per author"
                main_classifier.clf_custom_score(newlist_gold_gender, newlist_pred_gender, new_comment)

                new_comment = "age per author"
                main_classifier.clf_custom_score(newlist_gold_age, newlist_pred_age, new_comment)

            else:
                truth_file_tmp = get_truth(options.input, options.input, lang)
                result_dict = get_results(options.input, lang)

                # filter truth dictionary to only include the evaluated authors

                truth_file = dict()
                for element in truth_file_tmp:
                    # print element
                    if element in result_dict:
                        truth_file[element] = truth_file_tmp[element]

                newlist_gold_gender = list()
                for key in sorted(truth_file):
                    newlist_gold_gender.append(truth_file[key][0])

                newlist_pred_gender = list()
                for key in sorted(truth_file):
                    newlist_pred_gender.append(result_dict[key][0])

                # print truth_file

                new_comment = "gender per author"
                main_classifier.clf_custom_score(newlist_gold_gender, newlist_pred_gender, new_comment)


if __name__ == '__main__':
    main(sys.argv)
    sys.exit(1)
