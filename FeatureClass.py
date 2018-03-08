# -*- coding: utf-8 -*-

__version__ = "1.0"
__date__ = "24.07.2016"
__author__ = "Ivan Bilan"

import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from time import sleep
import codecs
import re
import numpy

class FeatureClass(object):

    def __init__(self, lang):

        # load resources

        self.language = lang
        if lang == "en":

            self.connective_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources/en/connective_words.txt'))
            self.slang_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\en\\slang_words.txt'))
            # get emoticon words
            # list taken from https://gist.github.com/ryanlewis/a37739d710ccdb4b406d
            self.emotion_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\en\\emotion_words.txt'))
            # get swear words
            self.abbreviation_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\en\\abbreviation.txt'))
            self.contractions_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\en\\contractions.txt'))
            self.cachedStopWords = stopwords.words("english")
        elif lang == "nl":
            # get connective words
            self.connective_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\nl\\connective_words.txt'))

            self.slang_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\nl\\slang_words.txt'))
            # get emoticon words
            # list taken from https://gist.github.com/ryanlewis/a37739d710ccdb4b406d
            self.emotion_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\nl\\emotion_words.txt'))
            # get swear words
            # self.swear_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\nl\\swear_words.txt')
            self.abbreviation_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\nl\\abbreviation.txt'))
            self.contractions_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\nl\\contractions.txt'))
            self.cachedStopWords = stopwords.words("dutch")
        elif lang == "es":
            # get connective words
            self.connective_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\es\\connective_words.txt'))
            self.slang_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\es\\slang_words.txt'))
            # get emoticon words
            self.emotion_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\es\\emotion_words.txt'))
            # get swear words
            # self.swear_words_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\es\\swear_words.txt')
            self.abbreviation_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\es\\abbreviation.txt'))
            self.contractions_list = self.extract_bow(os.path.join(os.path.dirname(__file__), 'Resources\\es\\contractions.txt'))
            self.cachedStopWords = stopwords.words("spanish")

        # emoticon regex
        self.m_emoticons_comp = re.compile("[\;\:\=]\-*[\)\(\]\>\/]")

        # pre-scaling text levels
        calc = list()
        self.calc_words = self.recursive_addition(calc, 0, 13, 10450)
        calc2 = list()
        self.calc_chars = self.recursive_addition(calc2, 0, 80, 64800)

    def recursive_addition(self, list_inner, val, step, limit):
        val+=step
        list_inner.append(val)

        if val < limit:
            return self.recursive_addition(list_inner, val, step, limit)
        else:
            return list_inner

    def regex_str(self, items):
        """
        Create a regex out of a list
        """
        new_list = list()
        for x in items:
            # print x
            new_list.append(re.escape(x))

        fulls_joined = '|'.join(new_list)
        my_regex = r'(^|\b)('+fulls_joined+r')(\s|$)'  # repr(fulls_joined)
        # print my_regex
        return my_regex

    def extract_bow(self, filename):
        """
        Extract tokens from txt file
        """

        input_file = codecs.open(filename, encoding = "utf_8", mode='r')
        lines = input_file.readlines()
        bow_words = []
        for x in lines:
            x = x.replace('\r\n', '')
            bow_words.append(x.strip().lower())
        input_file.close()
        return list(set(bow_words))

    def catch_url(self, untagged, average_value = 0):
        """
        Get the usage of linked content
        """
        if average_value == 3:
            sum_vector = 0
            for single_text in untagged:
                tokens = single_text.split()
                result = re.findall("\[URL\]", untagged)

                if (len(result) > 0) and (len(tokens) > 1):
                    average = float(len(result)) / len(tokens)
                    sum_vector += average
                else:
                    average = 0
                    sum_vector += average

            if len(untagged) > 0:
                average_return = sum_vector / len(untagged)
            else:
                average_return = sum_vector

            return average_return

        else:
            tokens = untagged.split()
            result = re.findall(r"url", untagged, re.IGNORECASE)

            if average_value == 0:
                return len(result)
            elif average_value == 1:
                if (len(result) > 0) and (len(tokens) > 0):
                    average = float(len(result)) / len(tokens)
                else:
                    average = 0

                return average

            elif average_value == 5:
                return self.counter_pre_scaling(len(result), len(word_tokenize(untagged)))

    def contractions(self, untagged, average_value = 0):
        # count the amount of contractions in the text
        # list of words taken from http://www.textfixer.com/resources/english-contractions-list.php
        return self.word_counter_feature(self.contractions_list, untagged, average_value)

    '''
    # not included, may be used in future work
    # this function counts the amount of swear words in the blog posts
    def swear_words(self, untagged, average_value = 5):
        # list taken from https://gist.github.com/ryanlewis/a37739d710ccdb4b406d
        return self.word_counter_feature(self.swear_words_list, untagged, average_value)
    '''

    def emotion_words(self, untagged, average_value = 5):
        # this function counts the amount of emotion words ( e.g. disgusted, hurt, aggressive )
        # http://www.psychpage.com/learning/library/assess/feelings.html
        return self.word_counter_feature(self.emotion_words_list, untagged, average_value)

    def slang_words(self, untagged, average_value = 5):
        # this function counts the amount of slang words
        # http://www.psychpage.com/learning/library/assess/feelings.html
        return self.word_counter_feature(self.slang_words_list, untagged, average_value)

    def connective_words(self, untagged, average_value = 5):
        #### checked
        # feature suggested here : http://www.uni-weimar.de/medien/webis/research/events/pan-13/pan13-talks/pan13-author-profiling/meina13-poster.pdf
        # these feature looks for all connective words (words that provide stylistic connection between sentences, paragraphs etc.)
        # http://www.grammarbank.com/connectives-list.html
        return self.word_counter_feature(self.connective_words_list, untagged, average_value)

    def get_abbreviations(self, untagged, average_value = 5):
        # http://www.grammarbank.com/connectives-list.html
        # list taken from http://www.textfixer.com/resources/english-contractions-list.php
        return self.word_counter_feature(self.abbreviation_list, untagged, average_value)

    # get all emoticons
    # not used, use for future work
    def get_emoticons(self, untagged, average_value = 0):
        if average_value == 3:

            sum_vector = 0
            for single_text in untagged:

                counter = 0

                # tokens = word_tokenize(untagged)
                tokens = single_text.split()

                for token in tokens:
                    m_emoticons = self.m_emoticons_comp.match(token)
                    if m_emoticons:
                        counter += 1
                        #print token

                if (counter > 0) and (len(tokens) > 1):
                    average = float(counter) / len(tokens)
                    sum_vector += average
                else:
                    average = 0
                    sum_vector += average
            if len(untagged) > 0:
                average_return = sum_vector / len(untagged)
            else:
                average_return = sum_vector
            return average_return
        else:
            counter = 0
            tokens = untagged.split()

            for token in tokens:
                m_emoticons = self.m_emoticons_comp.match(token)
                if m_emoticons:
                    counter += 1
            if average_value == 0:
                return counter
            elif average_value == 1:
                if (counter > 0) and (len(tokens) > 1):
                    average = float(counter) / len(tokens)
                else:
                    average = 0

                return average
        # get all emoticons

    def positive_emoticons(self, untagged, average_value=0):
        # not used
        # this function counts the amount of positive emoticons
        if average_value == 3:

            sum_vector = 0
            for single_text in untagged:
                # print single_text, len(single_text)
                counter = 0
                # tokens = word_tokenize(untagged)
                tokens = single_text.split()

                list_of_unicode_smilies = [u'😀', u'😁', u'😂', u'😃',u'😄',u'😅',u'😆', u'😉',u'😊',u'😋',  u'😎',u'🙋',u'😸', u'😛',u'?']
                for token in tokens:
                    # m = re.match("(\:|\;)(\=|c|\-\o)*(\]|\)|\D|\*)*", token)
                    m = re.match("((?::|;|<)(?:-|,)?(?:\)|D|3))", token)
                    if m: # or m2
                        #print token
                        counter += 1
                    if token in list_of_unicode_smilies:
                       #print token
                       counter += 1

                if (counter > 0) and (len(tokens) > 1):
                    average = float(counter) / len(tokens)
                    sum_vector += average
                else:
                    average = 0
                    sum_vector += average
            if len(untagged) > 0:
                average_return = sum_vector / len(untagged)
            else:
                average_return = sum_vector
            # print average_return
            return average_return
        else:
            counter = 0
            tokens = untagged.split()

            list_of_unicode_smilies = [u'😀', u'😁', u'😂', u'😃',u'😄',u'😅',u'😆', u'😉',u'😊',u'😋',  u'😎',u'🙋',u'😸', u'😛',u'?']
            for token in tokens:
                # m = re.match("(\:|\;)(\=|c|\-\o)*(\]|\)|\D|\*)*", token)
                m = re.match("((?::|;|<)(?:-|,)?(?:\)|D|3))", token)
                if m: # or m2
                    #print token
                    counter += 1
                if token in list_of_unicode_smilies:
                   #print token
                   counter += 1

            if average_value == 0:
                return counter
            elif average_value == 1:
                if (counter > 0) and (len(tokens) > 3):
                    average = float(counter) / len(tokens)
                else:
                    average = 0

                return average

    # not used
    # this function counts the amount of negative emoticons
    def negative_emoticons(self, untagged, average_value = 0):

        if average_value == 3:

            sum_vector = 0
            for single_text in untagged:

                counter = 0
                tokens = single_text.split()
                #print tokens[0]
                list_of_unicode_smilies = [u'😒', u'😕', u'😟', u'😠', u'😞', u'😢', u'😦', u'😧', u'😬', u'😿', u'🙎']

                for token in tokens:
                    # m = re.match("(\:|\;)(\=|c|\-\o)*(\]|\)|\D)*", token)
                    #m = re.match("((?::|;|=|D)(?:-)?(?:\(|\\|x|X|8|c|\[|\:))", token) # |\:
                    m = re.match("((?::|;|=|D)(?:-)?(?:\(|\\|x|X|8|c|C|\[|\:))", token) # |\:
                    #m2 = re.match("\<\3*", token)
                    if m : # or m2
                        #print token
                        counter += 1
                    if token in list_of_unicode_smilies:
                       #print token
                       counter += 1

                if (counter > 0) and (len(tokens) > 1):
                    average = float(counter) / len(tokens)
                    sum_vector += average
                else:
                    average = 0
                    sum_vector += average

            if len(untagged) > 0:
                average_return = sum_vector / len(untagged)
            else:
                average_return = sum_vector

            return average_return
        else:

            counter = 0
            tokens = untagged.split()
            #print tokens[0]
            list_of_unicode_smilies = [u'😒', u'😕', u'😟', u'😠', u'😞', u'😢', u'😦', u'😧', u'😬', u'😿', u'🙎']

            for token in tokens:
                # m = re.match("(\:|\;)(\=|c|\-\o)*(\]|\)|\D)*", token)
                #m = re.match("((?::|;|=|D)(?:-)?(?:\(|\\|x|X|8|c|\[|\:))", token) # |\:
                m = re.match("((?::|;|=|D)(?:-)?(?:\(|\\|x|X|8|c|C|\[|\:))", token) # |\:
                #m2 = re.match("\<\3*", token)
                if m : # or m2
                    #print token
                    counter += 1
                if token in list_of_unicode_smilies:
                    #print token
                    counter += 1
            if average_value == 0:
                return counter
            elif average_value == 1:
                if (counter > 0) and (len(tokens) > 3):
                    average = float(counter) / len(tokens)
                else:
                    average = 0

                return average

    # not used
    # this function counts the amount of neutral emoticons
    def neutral_emoticons(self, untagged, average_value = 0):
        if average_value == 3:

            sum_vector = 0
            for single_text in untagged:

                counter = 0
                list_of_unicode_smilies = [u'😐', u'😑', u'😶']
                tokens = single_text.split()
                #print tokens[0]

                for token in tokens:
                    # m = re.match("(\:|\;)(\=|c|\-\o)*(\]|\)|\D|\*)*", token)
                    m = re.match("((?::|=|<)(?:-)?(?:\||o|O))", token)
                    # m2 = re.match("\<\3*", token)
                    if m:
                        counter += 1
                    if token in list_of_unicode_smilies:
                       counter += 1

                if (counter > 0) and (len(tokens) > 1):
                    average = float(counter) / len(tokens)
                    sum_vector += average
                else:
                    average = 0
                    sum_vector += average

            if len(untagged) > 0:
                average_return = sum_vector / len(untagged)
            else:
                average_return = sum_vector

            return average_return
        else:
            counter = 0
            list_of_unicode_smilies = [u'😐', u'😑', u'😶']
            tokens = untagged.split()
            #print tokens[0]

            for token in tokens:
                # m = re.match("(\:|\;)(\=|c|\-\o)*(\]|\)|\D|\*)*", token)
                m = re.match("((?::|=|<)(?:-)?(?:\||o|O))", token)
                # m2 = re.match("\<\3*", token)
                if m:
                    counter += 1

                if token in list_of_unicode_smilies:
                    counter += 1
            if average_value == 0:
                return counter
            elif average_value == 1:
                if (counter > 0) and (len(tokens) > 3):
                    average = float(counter) / len(tokens)
                else:
                    average = 0

                return average

    # not used
    def quotation(untagged):
        counter = 0
        # regex to find all quoted content
        prog = re.compile(r'''".*?"''')
        # count all occurrences of quoted content
        counter = len(prog.findall(untagged))

        # test
        #if counter != 0 :
        #    print prog.findall(conversation)
        #    print counter

        return counter

    # count all punctuation marks, without special characters
    # we use a list of unicode characters because this
    # solution works better than with 'string.punctuation'
    def general_punctuation_new(self, untagged, average_value=0):
        if average_value == 3:
            pass
        else:
            counter = 0
            #print conversation

            count = lambda l1,l2: sum([1 for x in l1 if x in l2])
            counter = count(untagged,set(punctuation))

            '''
            if counter > 2:
                print untagged
                print counter
            '''

            if average_value == 0:
                # print counter
                return counter

            elif average_value == 1:
                if len(untagged) > 0:
                    result = float(counter) / len(untagged.split())
                    # print result
                    return result
                else:
                    return 0
            elif average_value == 5:
                return self.counter_pre_scaling_char(counter, len(untagged))

    def general_punctuation(self, untagged, average_value=0):
        if average_value == 3:
            pass
        else:


            list_of_punct = [u"\u0021", # exclamation mark
                             u"\u002E", # fullstop
                             u"\u002D", # hyphen
                             u"\u003B", # semicolon
                             u"\u0337", u"\u0338", u"\u002F",u"\u005C", # slash, solidus
                             u"\u003F" , # question mark
                             u"\u005B" ,u"\u005D" ,u"\u007B",u"\u007D",u"\u0028",u"\u0029", #brackets
                             u"\u0084" ,u"\u0022", u"\u00BB", u"\u00AB" # quotation marks
                             u"\u2024" ,u"\u2025" ,u"\u2026", # ellipsis
                             u"\u2012" ,u"\u2013" ,u"\u2014" ,u"\u2015" , # dash
                             u"\u2018" ,u"\u2019" ,u"\u2020" ,u"\u201A" ,u"\u201B" , # quotation line 1
                             u"\u201C" ,u"\u201D" ,u"\u201E" ,u"\u201F" ,# quotation
                             u"\u003A" , # colon
                             u"\u002C", u"\u02BD",u"\u02BB", u"\u0312",u"\u0313",u"\u0314",u"\u0315",# comma
                             u"\u02BC" ,u"\u0027",# apostrophe
                             u"\u02EE" , #double apostrophe
                            # special characters
                            u"\u2010" , u"\u2011" , u"\u2012", u"\u2013", u"\u2014",u"\u2015",
                            u"\u2016", u"\u2017",u"\u2018",u"\u2020",u"\u201A", u"\u201B",
                           u"\u201C", u"\u201D", u"\u201E", u"\u201F", u"\u2020", u"\u2021",
                           u"\u2022", u"\u2023", u"\u2024", u"\u2025", u"\u2026", u"\u2027",
                           u"\u2028", u"\u2029", u"\u2030", u"\u2031", u"\u2032", u"\u2033",
                           u"\u2034", u"\u2035", u"\u2036", u"\u2037", u"\u2038", u"\u2039",
                           u"\u2040", u"\u2041", u"\u2042", u"\u2043", u"\u2044", u"\u2045",
                           u"\u2046", u"\u2047", u"\u2048", u"\u2049", u"\u2050", u"\u2051",
                           u"\u2052", u"\u2053", u"\u2054", u"\u2055", u"\u2056", u"\u2057",
                           u"\u2058", u"\u2059", u"\u2060", u"\u2061", u"\u2062", u"\u2063",
                           u"\u2064", u"\u2065", u"\u2066", u"\u2067", u"\u2068", u"\u2069",
                           u"\u206A", u"\u206B", u"\u206C", u"\u206D", u"\u206E", u"\u206F",
                           u"\u205A", u"\u205B", u"\u205C", u"\u205D", u"\u205E", u"\u205F",
                           u"\u204A", u"\u204B", u"\u204C", u"\u204D", u"\u204E", u"\u204F",
                           u"\u203A", u"\u203B", u"\u203C", u"\u203D", u"\u203E", u"\u203F",
                           u"\u202A", u"\u202B", u"\u202C", u"\u202D", u"\u202E", u"\u202F"
                            ]
            counter = 0

            for x in untagged:
                for char in x:
                    # print char
                    if char in list_of_punct:
                        counter += 1

            if average_value == 0:
                # print counter
                return counter

            elif average_value == 1:
                if len(untagged) > 0:
                    result = float(counter) / len(untagged)
                    # print result
                    return result
                else:
                    return 0

    def special_characters(self, untagged):
        # get all special characters, that are not included in the general_punctuation feature
        list_of_punct = [u"\u2010" , u"\u2011" , u"\u2012", u"\u2013", u"\u2014",u"\u2015",
                          u"\u2016", u"\u2017",u"\u2018",u"\u2020",u"\u201A", u"\u201B",
                           u"\u201C", u"\u201D", u"\u201E", u"\u201F", u"\u2020", u"\u2021",
                           u"\u2022", u"\u2023", u"\u2024", u"\u2025", u"\u2026", u"\u2027",
                           u"\u2028", u"\u2029", u"\u2030", u"\u2031", u"\u2032", u"\u2033",
                           u"\u2034", u"\u2035", u"\u2036", u"\u2037", u"\u2038", u"\u2039",
                           u"\u2040", u"\u2041", u"\u2042", u"\u2043", u"\u2044", u"\u2045",
                           u"\u2046", u"\u2047", u"\u2048", u"\u2049", u"\u2050", u"\u2051",
                           u"\u2052", u"\u2053", u"\u2054", u"\u2055", u"\u2056", u"\u2057",
                           u"\u2058", u"\u2059", u"\u2060", u"\u2061", u"\u2062", u"\u2063",
                           u"\u2064", u"\u2065", u"\u2066", u"\u2067", u"\u2068", u"\u2069",
                           u"\u206A", u"\u206B", u"\u206C", u"\u206D", u"\u206E", u"\u206F",
                           u"\u205A", u"\u205B", u"\u205C", u"\u205D", u"\u205E", u"\u205F",
                           u"\u204A", u"\u204B", u"\u204C", u"\u204D", u"\u204E", u"\u204F",
                           u"\u203A", u"\u203B", u"\u203C", u"\u203D", u"\u203E", u"\u203F",
                           u"\u202A", u"\u202B", u"\u202C", u"\u202D", u"\u202E", u"\u202F"
                           ]
        counter = 0
        for x in untagged:
            if x in list_of_punct:
                counter += 1
                #print x
        #print counter
        return counter

    ##################### Gender Preferential Features START #############################
    def stylistic_ending_custom(self, untagged, average_value = 0, custom_ending=None):
        # count all words that end with -able, -ful, -al, -ible, -ic, -ive, -less, -ous
        ## these features are introduced in http://www.aclweb.org/anthology/D10-1021
        if average_value == 3:
            sum_value = 0

            for text in untagged:
                # tokens = word_tokenize(untagged)
                tokens = text.split()

                counter = 0
                for x in tokens:
                    if x.endswith(custom_ending):
                        counter += 1
                    sum_value += counter

            if len(untagged) > 0:
                average_inner = sum_value / len(untagged)
            else:
                average_inner = sum_value
            return average_inner

        else:
            tokens = word_tokenize(untagged)

            '''
            counter = 0
            for x in tokens:
                if x.endswith(custom_ending):
                    counter += 1
            '''
            my_regex = r'(^|\b)(\w+'+custom_ending+r')(\s|$)'
            custom_regex = re.compile(my_regex, re.IGNORECASE)
            counter = len(custom_regex.findall(untagged))

            if average_value == 0:
                # print counter
                return counter

            elif average_value == 1:
                if (counter > 0) and (len(tokens) > 0):
                    average = float(counter) / len(tokens)
                else:
                    average = 0

                # print average
                return average
            elif average_value == 5:
                return self.counter_pre_scaling(counter, len(tokens))
    ##################### Gender Preferential Features END #############################

    # calculate token/type ratio
    def type_token_ratio(self, untagged, average_value = 0):
        if average_value == 3:
            sum_value = 0
            for text in untagged:
                # tokenize the conversation
                # tokens = word_tokenize(text)
                tokens = text.split()
                relevant_tokens = []
                for x in tokens:
                    if len(x) > 2:
                        relevant_tokens.append(x)
                types = set(relevant_tokens)
                if (len(types)) != 0:
                    ratio = float(float(len(types)) / float(len(relevant_tokens)) * 100)
                else:
                    ratio = 0
                sum_value += ratio
            if len(untagged) > 0:
                average_inner = sum_value / len(untagged)
                return average_inner
            else:
                return sum_value
        else:
            # tokenize the conversation
            tokens = word_tokenize(untagged)
            relevant_tokens = []
            for x in tokens:
                if len(x) > 2:
                    relevant_tokens.append(x)
            types = set(relevant_tokens)
            if (len(types)) != 0:
                ratio = float(float(len(types)) / float(len(relevant_tokens)) * 100)
            else:
                ratio = 0
            return ratio

    def amount_of_tokens(self, untagged):
        splited_conversation = untagged.split()
        words = []
        for x in splited_conversation:
            x = re.sub('[\.!\?\,\:\'\"]', '', x)
            if (len(x) >= 1):
                words.append(x)
        #print len(splited_conversation)
        return len(words)

    def amount_of_types(self, untagged):
        splited_conversation = untagged.split()
        words = []
        for x in splited_conversation:
            x = re.sub('[\.!\?\,\:\'\"]', '', x)
            if (len(x) >= 1):
                words.append(x)
        #print len(splited_conversation)
        set_words = set(words)
        return len(set_words)

    # not used
    # this feature was suggested in:
    # 'Forensic Psycholinguistics. Using Language Analysis for Identifying and Assessing Offenders'
    # http://diogenesllc.com/statementlinguistics.pdf
    def amount_sorryWords(self, untagged, average_value=0):
        if average_value == 3:
            pass
        else:
            tokens = untagged.split()
            result = re.findall(r"sorry", untagged)

            if average_value == 0:
                # print len(result)
                return len(result)
            elif average_value == 1:
                if (len(result) > 0) and (len(tokens) > 0):
                    average = float(len(result)) / len(tokens)
                else:
                    average = 0
                # print average
                return average

    def average_wordlength(self, untagged, average_value=0):
        # calculate average word length
        if average_value == 3:
            # add funct for PAN
            pass
        else:
            if len(untagged.split()) == 0:
                average_word_length = 0
            else:
                average_word_length = numpy.mean([len(word) for word in untagged.split()])
            if isinstance(average_word_length, float):
                pass
            else:
                average_word_length = 0
            return average_word_length

    def words_capitalized(self, untagged, average_value = 5):
        # count capitalized words

        if average_value == 3:
            sum_value = 0
            for text in untagged:
                # tokenize the conversation
                # tokens = word_tokenize(untagged)
                tokens = text.split()
                # amount_words = len(tokens)
                counter = 0
                for word in tokens:
                    if word[0].isupper():
                        counter += 1
                    sum_value += counter
            if len(untagged) > 0:
                average_inner = sum_value / len(untagged)
            else:
                average_inner = sum_value
            return average_inner
        else:
            # tokenize the conversation
            tokens = untagged.split()
            counter = 0
            for index, word in enumerate(tokens):
                # print word
                if word[0].isupper() and word != 'URL' and index != 0:
                    counter += 1

            if average_value == 0:
                # print counter
                return counter

            elif average_value == 1:
                if (counter > 0) and (len(tokens) > 0):
                    average = float(counter) / len(tokens)
                else:
                    average = 0
                # print average
                return average
            elif average_value == 5:
                return self.counter_pre_scaling(counter, len(tokens))

    def AllCaps(self, untagged, average_value=5):
        # counts words with all capital letters
        # tokenize the conversation
        tokens = word_tokenize(untagged)

        counter = 0
        for word in tokens:
            if word.isupper() and not word == 'URL' and not word == 'USER': # or [URL]
                counter += 1
                #print word

        if average_value == 0:
                # print counter
                return counter
        elif average_value == 1:
            if (counter > 0) and (len(tokens) > 0):
                average = float(counter) / len(tokens)
            else:
                average = 0

            # print average
            return average
        elif average_value == 5:
            return self.counter_pre_scaling(counter, len(tokens))

    ############## Features that require Pos_tags START #########################

    def count_stop_words(self, untagged, average_value = 0):
        # import nltk
        # nltk.download()
        # print len(stopwords.words('english'))
        return self.word_counter_feature(self.cachedStopWords, untagged, average_value)

    def each_part_of_speech(self, tagged, average_value=0, custom_pos=None):

        ### all POS tags are based on TreeTagger tags
        ### in TreeTagger, each language has a different TAG set
        if self.language == "en":
            if average_value == 3:
                pass
            else:
                splitted_text = tagged.split()

                if custom_pos == "nouns":
                    # pos_counter = int(splitted_text.count("NN") + splitted_text.count("NNP") + splitted_text.count("NNPS") + splitted_text.count("NNS"))
                    list_to_find = ["NN", "NNP", "NNPS", "NNS"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))

                elif custom_pos == "adjectives":
                    # pos_counter = int(splitted_text.count("JJ") + splitted_text.count("JJR") + splitted_text.count("JJS"))
                    list_to_find = ["JJ", "JJR", "JJS"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "determiner":
                    # pos_counter = int(splitted_text.count("DT") + splitted_text.count("WDT") + splitted_text.count("PDT"))
                    list_to_find = ["DT", "WDT", "PDT"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "conjunctions":
                    # pos_counter = int(splitted_text.count("CC") + splitted_text.count("IN"))
                    list_to_find = ["CC", "IN"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "pronouns":
                    # pos_counter = int(splitted_text.count("PRP") + splitted_text.count("PRP$") + splitted_text.count("WP") + splitted_text.count("WP$"))
                    list_to_find = ["PRP", "PRP$", "WP", "WP$"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "verbs":
                    # pos_counter = int(splitted_text.count("VB") + splitted_text.count("VBD") + splitted_text.count("VBG") + splitted_text.count("VBN") + splitted_text.count("VBP") + splitted_text.count("VBZ"))
                    list_to_find = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "adverbs":
                    # pos_counter = int(splitted_text.count("RB") + splitted_text.count("RBR") + splitted_text.count("RBS"))
                    list_to_find = ["RB", "RBR", "RBS"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "modals":
                    # pos_counter = int(splitted_text.count("MD"))
                    list_to_find = ["MD"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "interjections":
                    # pos_counter = int(splitted_text.count("UH"))
                    list_to_find = ["UH"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "to_pos":
                    # pos_counter = int(splitted_text.count("TO"))
                    list_to_find = ["TO"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "cardinal_num":
                    # pos_counter = int(splitted_text.count("CD"))
                    list_to_find = ["CD"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))

                if average_value == 0:
                    return pos_counter
                elif average_value == 1:
                    if len(splitted_text) > 0:
                        result = float(pos_counter) / len(splitted_text)
                    else:
                        result = 0
                    return result
                elif average_value == 5:
                    return self.counter_pre_scaling(pos_counter, len(splitted_text))

        elif self.language == "nl":
            if average_value == 3:
                pass
            else:

                splitted_text = tagged.split()

                if custom_pos == "nouns":
                    # pos_counter = int(splitted_text.count("NN") + splitted_text.count("NNP") + splitted_text.count("NNPS") + splitted_text.count("NNS"))
                    list_to_find = ["noun*kop", "nounabbr", "nounpl", "nounprop", "nounsg"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))

                elif custom_pos == "adjectives":
                    # pos_counter = int(splitted_text.count("JJ") + splitted_text.count("JJR") + splitted_text.count("JJS"))
                    list_to_find = ["adj", "adj*kop", "adjabbr"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "determiner":
                    # pos_counter = int(splitted_text.count("DT") + splitted_text.count("WDT") + splitted_text.count("PDT"))
                    list_to_find = ["det__demo", "prondemo", "det__art"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "conjunctions":
                    # pos_counter = int(splitted_text.count("CC") + splitted_text.count("IN"))
                    list_to_find = ["conjcoord", "conjsubo"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "pronouns":
                    # pos_counter = int(splitted_text.count("PRP") + splitted_text.count("PRP$") + splitted_text.count("WP") + splitted_text.count("WP$"))
                    list_to_find = ["pronindef", "pronpers", "pronposs", "pronquest", "pronrefl", "pronrel"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "verbs":
                    # pos_counter = int(splitted_text.count("VB") + splitted_text.count("VBD") + splitted_text.count("VBG") + splitted_text.count("VBN") + splitted_text.count("VBP") + splitted_text.count("VBZ"))
                    list_to_find = ["verbinf", "verbpapa", "verbpastpl", "verbpastsg", "verbpresp", "verbprespl", "verbpressg"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "adverbs":
                    # pos_counter = int(splitted_text.count("RB") + splitted_text.count("RBR") + splitted_text.count("RBS"))
                    list_to_find = ["adv", "advabbr", "pronadv"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "modals":
                    # pos_counter = int(splitted_text.count("MD"))
                    # here changed to particle -te in Dutch
                    list_to_find = ["partte"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "interjections":
                    # pos_counter = int(splitted_text.count("UH"))
                    list_to_find = ["int"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "to_pos":
                    # pos_counter = int(splitted_text.count("TO"))
                    list_to_find = ["prep", "prepabbr"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "cardinal_num":
                    # pos_counter = int(splitted_text.count("CD"))
                    list_to_find = ["num__card", "num__ord"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))

                if average_value == 0:
                    return pos_counter
                elif average_value == 1:
                    if len(splitted_text) > 0:
                        result = float(pos_counter) / len(splitted_text)
                    else:
                        result = 0
                    return result
                elif average_value == 5:
                    return self.counter_pre_scaling(pos_counter, len(splitted_text))

        elif self.language == "es":
            if average_value == 3:
                pass
            else:
                splitted_text = tagged.split()
                if custom_pos == "nouns":
                    # pos_counter = int(splitted_text.count("NN") + splitted_text.count("NNP") + splitted_text.count("NNPS") + splitted_text.count("NNS"))
                    list_to_find = ["NN", "NP", "NC", "NMEA", "PAL", "PDEL", "PE"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))

                elif custom_pos == "adjectives":
                    # pos_counter = int(splitted_text.count("JJ") + splitted_text.count("JJR") + splitted_text.count("JJS"))
                    list_to_find = ["ADJ"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))

                elif custom_pos == "determiner":
                    # pos_counter = int(splitted_text.count("DT") + splitted_text.count("WDT") + splitted_text.count("PDT"))
                    list_to_find = ["DM"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "conjunctions":
                    # pos_counter = int(splitted_text.count("CC") + splitted_text.count("IN"))
                    list_to_find = ["CC", "CCAD", "CQUE", "CSUBF", "CSUBI", "CSUBX"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "pronouns":
                    # pos_counter = int(splitted_text.count("PRP") + splitted_text.count("PRP$") + splitted_text.count("WP") + splitted_text.count("WP$"))
                    list_to_find = ["DM", "INT", "PPC", "PPO", "PPX", "REL"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "verbs":
                    # pos_counter = int(splitted_text.count("VB") + splitted_text.count("VBD") + splitted_text.count("VBG") + splitted_text.count("VBN") + splitted_text.count("VBP") + splitted_text.count("VBZ"))
                    list_to_find = ["VCLIger", "VCLIinf", "VCLIfin", "VEadj", "VEfin", "VEger", "VEinf", "VHadj", "VEadj", "VHfin", "VHger" , "VHinf", "VLadj", "VLfin", "VLger", "VLinf",  "VSadj", "VSfin", "VSger", "VSinf"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "adverbs":
                    # pos_counter = int(splitted_text.count("RB") + splitted_text.count("RBR") + splitted_text.count("RBS"))
                    list_to_find = ["ADV"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "modals":
                    # pos_counter = int(splitted_text.count("MD"))
                    list_to_find = ["VMadj", "VMfin", "VMger", "VMinf"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "interjections":
                    # pos_counter = int(splitted_text.count("UH"))
                    list_to_find = ["ITJN"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "to_pos":
                    # pos_counter = int(splitted_text.count("TO"))
                    # here prepositions
                    list_to_find = ["PREP", "PREP/DEL"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))
                elif custom_pos == "cardinal_num":
                    # pos_counter = int(splitted_text.count("CD"))
                    list_to_find = ["CARD"]
                    custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                    pos_counter = len(custom_regex.findall(tagged))

                if average_value == 0:
                    return pos_counter
                elif average_value == 1:
                    if len(splitted_text) > 0:
                        result = float(pos_counter) / len(splitted_text)
                    else:
                        result = 0
                    return result
                elif average_value == 5:
                    return self.counter_pre_scaling(pos_counter, len(splitted_text))

    def counter_pre_scaling_char(self, counter, text_length):
        '''
        in characters
        tweets: min 6, mean 80, max 320
        review: min 5, mean 410, max 16673
        blogs: min 1, mean 3058, max 64790
        '''

        for index, element in enumerate(self.calc_chars):
            if text_length < element:
                result = float(counter) / (index+1)
                break
            elif text_length > self.calc_chars[-1]:
                if len(self.calc_chars) > 0:
                    result = float(counter) / len(self.calc_chars)
                else:
                    result = float(counter)
                break
        return result

    def counter_pre_scaling(self, counter, text_length):
        '''
        in words
        tweets: min 2, mean 13, max 33
        review: min 3, mean 75, max 2955
        blogs: min 1, mean 508, max 10424
        '''
        for index, element in enumerate(self.calc_words):
            if text_length < element:
                result = float(counter) / (index+1)
                break
            elif text_length > self.calc_words[-1]:
                if len(self.calc_words) > 0:
                    result = float(counter) / len(self.calc_words)
                else:
                    result = float(counter)
                break
        return result

    def word_counter_feature(self, word_list, text_sample, average_value):
        # feature to count all occurrences of given words in a text sample
        if average_value == 3:
            # over all texts per author
            sum_vector = 0
            for single_text in text_sample:
                counter = 0
                tokens = word_tokenize(text_sample)
                # tokens = single_text.split()
                connective = re.compile(self.regex_str(word_list), re.IGNORECASE)
                counter = len(connective.findall(text_sample))
                if (counter > 0) and (len(tokens) > 1):
                    average = float(counter) / len(tokens)
                    sum_vector += average
                else:
                    average = 0
                    sum_vector += average
            if len(text_sample) > 0:
                average_return = sum_vector / len(text_sample)
            else:
                average_return = sum_vector
            return average_return

        else:
            if average_value == 4:
                feature_vector = list()
                tokens = word_tokenize(text_sample)
                for token1 in tokens:
                    for token2 in word_list:
                        if token1 == token2:
                            feature_vector.append(1)
                        else:
                            feature_vector.append(0)
                return feature_vector
            else:
                # print self.regex_str(word_list)
                connective = re.compile(self.regex_str(word_list), re.IGNORECASE)
                counter = len(connective.findall(text_sample))
                # print connective.findall(text_sample)
                if average_value == 0:
                    # print counter
                    return counter
                elif average_value == 1:
                    tokens = text_sample.split()
                    if (counter > 0) and (len(tokens) > 1):
                        average = float(counter) / len(tokens)
                    else:
                        average = 0
                    # print average
                    return average

                elif average_value == 5:
                    result = self.counter_pre_scaling(counter, len(text_sample.split()))
                    return result

    def lexical_Fmeasure_new(self, tagged):
        if self.language == "en":
            # This feature calculates how implicit or explicit the text is.
            # It is a unitary measure of text's relative contextuality in opposition to its formality.
            # Feature is suggested in this paper: http://www.aclweb.org/anthology/D10-1021
            # # liste von alle TAGS >> http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

            counter_nouns = 0
            counter_adj = 0
            counter_prep = 0
            counter_art = 0
            counter_pron = 0
            counter_verb = 0
            counter_adv = 0
            counter_int = 0
            # print tagged

            splitted_text = tagged.split()

            # counter_nouns = int(splitted_text.count("NN") + splitted_text.count("NNP") + splitted_text.count("NNPS") + splitted_text.count("NNS"))
            list_to_find_nouns = ["NN", "NNP", "NNPS", "NNS"]
            custom_regex = re.compile(self.regex_str(list_to_find_nouns), re.IGNORECASE)
            counter_nouns = len(custom_regex.findall(tagged))
            counter_nouns = self.counter_pre_scaling(counter_nouns, len(splitted_text))

            # counter_adj = int(splitted_text.count("JJ") + splitted_text.count("JJR") + splitted_text.count("JJS"))
            list_to_find_adj = ["JJ", "JJR", "JJS"]
            custom_regex = re.compile(self.regex_str(list_to_find_adj), re.IGNORECASE)
            counter_adj = len(custom_regex.findall(tagged))
            counter_adj = self.counter_pre_scaling(counter_adj, len(splitted_text))


            # counter_prep = int(splitted_text.count("DT") + splitted_text.count("WDT") + splitted_text.count("PDT"))
            list_to_find_art = ["DT"]
            custom_regex = re.compile(self.regex_str(list_to_find_art), re.IGNORECASE)
            counter_art = len(custom_regex.findall(tagged))
            counter_art = self.counter_pre_scaling(counter_art, len(splitted_text))

            # counter_prep = int(splitted_text.count("IN"))
            list_to_find_prep = ["IN"]
            custom_regex = re.compile(self.regex_str(list_to_find_prep), re.IGNORECASE)
            counter_prep = len(custom_regex.findall(tagged))
            counter_prep = self.counter_pre_scaling(counter_prep, len(splitted_text))

            # counter_pron = int(splitted_text.count("PRP") + splitted_text.count("PRP$") + splitted_text.count("WP") + splitted_text.count("WP$"))
            list_to_find_pron = ["PRP", "PRP$", "WP", "WP$"]
            custom_regex = re.compile(self.regex_str(list_to_find_pron), re.IGNORECASE)
            counter_pron = len(custom_regex.findall(tagged))
            counter_pron = self.counter_pre_scaling(counter_pron, len(splitted_text))

            # counter_verb = int(splitted_text.count("VB") + splitted_text.count("VBD") + splitted_text.count("VBG") + splitted_text.count("VBN") + splitted_text.count("VBP") + splitted_text.count("VBZ"))
            list_to_find_verbs = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
            custom_regex = re.compile(self.regex_str(list_to_find_verbs), re.IGNORECASE)
            counter_verb = len(custom_regex.findall(tagged))
            counter_verb = self.counter_pre_scaling(counter_verb, len(splitted_text))

            # counter_adv = int(splitted_text.count("RB") + splitted_text.count("RBR") + splitted_text.count("RBS"))
            list_to_find_adv = ["RB", "RBR", "RBS"]
            custom_regex = re.compile(self.regex_str(list_to_find_adv), re.IGNORECASE)
            counter_adv = len(custom_regex.findall(tagged))
            counter_adv = self.counter_pre_scaling(counter_adv, len(splitted_text))

            # counter_int = int(splitted_text.count("UH"))
            list_to_find_int = ["UH"]
            custom_regex = re.compile(self.regex_str(list_to_find_int), re.IGNORECASE)
            counter_int = len(custom_regex.findall(tagged))
            counter_int = self.counter_pre_scaling(counter_int, len(splitted_text))

            '''
            print "Nouns: ", counter_nouns
            print "Adj: ", counter_adj
            print "Prep: ", counter_prep
            print "Art: ", counter_art
            print "Pron: ", counter_pron
            print "Verb: ", counter_verb
            print "Adv: ", counter_adv
            print "Int: ", counter_int
            '''

            F = 0.5 * ((counter_nouns + counter_adj + counter_prep + counter_art) - (counter_pron + counter_verb + counter_adv + counter_int ) + 100)
            # print "Lexical F-Measure :", F
            return F

        elif self.language == "nl":
            # This feature calculates how implicit or explicit the text is.
            # It is a unitary measure of text's relative contextuality in opposition to its formality.
            # Feature is suggested in this paper: http://www.aclweb.org/anthology/D10-1021
            # # liste von alle TAGS >> http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

            counter_nouns = 0
            counter_adj = 0
            counter_prep = 0
            counter_art = 0
            counter_pron = 0
            counter_verb = 0
            counter_adv = 0
            counter_int = 0

            # print tagged

            splitted_text = tagged.split()

            # counter_nouns = int(splitted_text.count("NN") + splitted_text.count("NNP") + splitted_text.count("NNPS") + splitted_text.count("NNS"))
            list_to_find_nouns = ["noun*kop", "nounabbr", "nounpl", "nounprop", "nounsg"]
            custom_regex = re.compile(self.regex_str(list_to_find_nouns), re.IGNORECASE)
            counter_nouns = len(custom_regex.findall(tagged))
            counter_nouns = self.counter_pre_scaling(counter_nouns, len(splitted_text))

            # counter_adj = int(splitted_text.count("JJ") + splitted_text.count("JJR") + splitted_text.count("JJS"))
            list_to_find_adj = ["adj", "adj*kop", "adjabbr"]
            custom_regex = re.compile(self.regex_str(list_to_find_adj), re.IGNORECASE)
            counter_adj = len(custom_regex.findall(tagged))
            counter_adj = self.counter_pre_scaling(counter_adj, len(splitted_text))

            # counter_prep = int(splitted_text.count("DT") + splitted_text.count("WDT") + splitted_text.count("PDT"))
            list_to_find_art = ["det__demo", "prondemo", "det__art"]
            custom_regex = re.compile(self.regex_str(list_to_find_art), re.IGNORECASE)
            counter_art = len(custom_regex.findall(tagged))
            counter_art = self.counter_pre_scaling(counter_art, len(splitted_text))

            # counter_prep = int(splitted_text.count("IN"))
            list_to_find_prep = ["prep", "prepabbr"]
            custom_regex = re.compile(self.regex_str(list_to_find_prep), re.IGNORECASE)
            counter_prep = len(custom_regex.findall(tagged))
            counter_prep = self.counter_pre_scaling(counter_prep, len(splitted_text))

            # counter_pron = int(splitted_text.count("PRP") + splitted_text.count("PRP$") + splitted_text.count("WP") + splitted_text.count("WP$"))
            list_to_find_pron = ["pronindef", "pronpers", "pronposs", "pronquest", "pronrefl", "pronrel"]
            custom_regex = re.compile(self.regex_str(list_to_find_pron), re.IGNORECASE)
            counter_pron = len(custom_regex.findall(tagged))
            counter_pron = self.counter_pre_scaling(counter_pron, len(splitted_text))

            # counter_verb = int(splitted_text.count("VB") + splitted_text.count("VBD") + splitted_text.count("VBG") + splitted_text.count("VBN") + splitted_text.count("VBP") + splitted_text.count("VBZ"))
            list_to_find_verbs = ["verbinf", "verbpapa", "verbpastpl", "verbpastsg", "verbpresp", "verbprespl", "verbpressg"]
            custom_regex = re.compile(self.regex_str(list_to_find_verbs), re.IGNORECASE)
            counter_verb = len(custom_regex.findall(tagged))
            counter_verb = self.counter_pre_scaling(counter_verb, len(splitted_text))

            # counter_adv = int(splitted_text.count("RB") + splitted_text.count("RBR") + splitted_text.count("RBS"))
            list_to_find_adv = ["adv", "advabbr", "pronadv"]
            custom_regex = re.compile(self.regex_str(list_to_find_adv), re.IGNORECASE)
            counter_adv = len(custom_regex.findall(tagged))
            counter_adv = self.counter_pre_scaling(counter_adv, len(splitted_text))

            # counter_int = int(splitted_text.count("UH"))
            list_to_find_int = ["int"]
            custom_regex = re.compile(self.regex_str(list_to_find_int), re.IGNORECASE)
            counter_int = len(custom_regex.findall(tagged))
            counter_int = self.counter_pre_scaling(counter_int, len(splitted_text))

            '''
            print "Nouns: ", counter_nouns
            print "Adj: ", counter_adj
            print "Prep: ", counter_prep
            print "Art: ", counter_art
            print "Pron: ", counter_pron
            print "Verb: ", counter_verb
            print "Adv: ", counter_adv
            print "Int: ", counter_int
            '''

            F = 0.5 * ((counter_nouns + counter_adj + counter_prep + counter_art) - (counter_pron + counter_verb + counter_adv + counter_int ) + 100)
            # print "Lexical F-Measure :", F
            return F

        elif self.language == "es":
            # This feature calculates how implicit or explicit the text is.
            # It is a unitary measure of text's relative contextuality in opposition to its formality.
            # Feature is suggested in this paper: http://www.aclweb.org/anthology/D10-1021
            # # liste von alle TAGS >> http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

            counter_nouns = 0
            counter_adj = 0
            counter_prep = 0
            counter_art = 0
            counter_pron = 0
            counter_verb = 0
            counter_adv = 0
            counter_int = 0

            # print tagged

            splitted_text = tagged.split()

            # counter_nouns = int(splitted_text.count("NN") + splitted_text.count("NNP") + splitted_text.count("NNPS") + splitted_text.count("NNS"))
            list_to_find_nouns = ["NN", "NP", "NC", "NMEA", "PAL", "PDEL", "PE"]
            custom_regex = re.compile(self.regex_str(list_to_find_nouns), re.IGNORECASE)
            counter_nouns = len(custom_regex.findall(tagged))
            counter_nouns = self.counter_pre_scaling(counter_nouns, len(splitted_text))

            # counter_adj = int(splitted_text.count("JJ") + splitted_text.count("JJR") + splitted_text.count("JJS"))
            list_to_find_adj = ["ADJ"]
            custom_regex = re.compile(self.regex_str(list_to_find_adj), re.IGNORECASE)
            counter_adj = len(custom_regex.findall(tagged))
            counter_adj = self.counter_pre_scaling(counter_adj, len(splitted_text))


            # counter_prep = int(splitted_text.count("DT") + splitted_text.count("WDT") + splitted_text.count("PDT"))
            list_to_find_art = ["DM"]
            custom_regex = re.compile(self.regex_str(list_to_find_art), re.IGNORECASE)
            counter_art = len(custom_regex.findall(tagged))
            counter_art = self.counter_pre_scaling(counter_art, len(splitted_text))

            # counter_prep = int(splitted_text.count("IN"))
            list_to_find_prep = ["PREP", "PREP/DEL"]
            custom_regex = re.compile(self.regex_str(list_to_find_prep), re.IGNORECASE)
            counter_prep = len(custom_regex.findall(tagged))
            counter_prep = self.counter_pre_scaling(counter_prep, len(splitted_text))

            # counter_pron = int(splitted_text.count("PRP") + splitted_text.count("PRP$") + splitted_text.count("WP") + splitted_text.count("WP$"))
            list_to_find_pron = ["DM", "INT", "PPC", "PPO", "PPX", "REL"]
            custom_regex = re.compile(self.regex_str(list_to_find_pron), re.IGNORECASE)
            counter_pron = len(custom_regex.findall(tagged))
            counter_pron = self.counter_pre_scaling(counter_pron, len(splitted_text))

            # counter_verb = int(splitted_text.count("VB") + splitted_text.count("VBD") + splitted_text.count("VBG") + splitted_text.count("VBN") + splitted_text.count("VBP") + splitted_text.count("VBZ"))
            list_to_find_verbs = ["VCLIger", "VCLIinf", "VCLIfin", "VEadj", "VEfin", "VEger", "VEinf", "VHadj", "VEadj", "VHfin", "VHger" , "VHinf", "VLadj", "VLfin", "VLger", "VLinf",  "VSadj", "VSfin", "VSger", "VSinf", "VMadj", "VMfin", "VMger", "VMinf"]
            custom_regex = re.compile(self.regex_str(list_to_find_verbs), re.IGNORECASE)
            counter_verb = len(custom_regex.findall(tagged))
            counter_verb = self.counter_pre_scaling(counter_verb, len(splitted_text))

            # counter_adv = int(splitted_text.count("RB") + splitted_text.count("RBR") + splitted_text.count("RBS"))
            list_to_find_adv = ["ADV"]
            custom_regex = re.compile(self.regex_str(list_to_find_adv), re.IGNORECASE)
            counter_adv = len(custom_regex.findall(tagged))
            counter_adv = self.counter_pre_scaling(counter_adv, len(splitted_text))

            # counter_int = int(splitted_text.count("UH"))
            list_to_find_int = ["ITJN"]
            custom_regex = re.compile(self.regex_str(list_to_find_int), re.IGNORECASE)
            counter_int = len(custom_regex.findall(tagged))
            counter_int = self.counter_pre_scaling(counter_int, len(splitted_text))

            '''
            print "Nouns: ", counter_nouns
            print "Adj: ", counter_adj
            print "Prep: ", counter_prep
            print "Art: ", counter_art
            print "Pron: ", counter_pron
            print "Verb: ", counter_verb
            print "Adv: ", counter_adv
            print "Int: ", counter_int
            '''

            F = 0.5 * ((counter_nouns + counter_adj + counter_prep + counter_art) - (counter_pron + counter_verb + counter_adv + counter_int ) + 100)
            # print "Lexical F-Measure :", F
            return F

    def plurality(self, tagged, average_value = 0):
        # This feature calculates the amount of Plural Nouns
        # this feature is suggested in http://ischool.syr.edu/media/documents/2013/2/beiyujan2013.pdf

        if self.language == "en":
            if average_value == 3:
                pass

            else:
                counter = 0
                splitted_text = tagged.split()

                # counter = int(splitted_text.count("NNS") + splitted_text.count("NNPS"))
                list_to_find = ["NNS", "NNPS"]
                custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                counter = len(custom_regex.findall(tagged))

                # print "Plurality: ", counter
                # print tagged.split()

                if average_value == 0:
                    return counter
                elif average_value == 1:
                    if (len(splitted_text) > 0) and (counter > 0):
                        counter_new = float(counter)/len(splitted_text)
                    else:
                        counter_new = 0

                    # test output
                    '''
                    if counter_new > 0:
                        print tagged
                        print counter
                        print counter_new
                    '''

                    return counter_new
                elif average_value == 5:
                    return self.counter_pre_scaling(counter, len(splitted_text))
        if self.language == "nl":
            if average_value == 3:
                pass

            else:
                counter = 0
                splitted_text = tagged.split()

                # counter = int(splitted_text.count("NNS") + splitted_text.count("NNPS"))
                list_to_find = ["nounpl", "verbpastpl", "verbprespl"]
                custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                counter = len(custom_regex.findall(tagged))

                # print "Plurality: ", counter
                # print tagged.split()

                if average_value == 0:
                    return counter
                elif average_value == 1:
                    if (len(splitted_text) > 0) and (counter > 0):
                        counter_new = float(counter)/ len(splitted_text)
                    else:
                        counter_new = 0

                    # test output
                    '''
                    if counter_new > 0:
                        print tagged
                        print counter
                        print counter_new
                    '''

                    return counter_new
                elif average_value == 5:
                    return self.counter_pre_scaling(counter, len(splitted_text))
        if self.language == "es":
            if average_value == 3:
                pass

            else:
                counter = 0
                splitted_text = tagged.split()
                # counter = int(splitted_text.count("NNS") + splitted_text.count("NNPS"))
                # here modified to:
                # ALFS	Singular letter of the alphabet (A, b)
                # SE	Se (as particle)
                list_to_find = ["ALFP", "SE"]
                custom_regex = re.compile(self.regex_str(list_to_find), re.IGNORECASE)
                counter = len(custom_regex.findall(tagged))

                # print "Plurality: ", counter
                # print tagged.split()

                if average_value == 0:
                    return counter
                elif average_value == 1:
                    if (len(splitted_text) > 0) and (counter > 0):
                        counter_new = float(counter)/ len(splitted_text)
                    else:
                        counter_new = 0

                    return counter_new
                elif average_value == 5:
                    return self.counter_pre_scaling(counter, len(splitted_text))

    # count all vowels
    def find_vowels(self, untagged):
        from collections import Counter
        vowels = "aeiuoAEIOU"
        c = Counter(untagged)

        return sum(c[v] for v in vowels)