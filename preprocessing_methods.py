# general
import os
import re
import inspect
import itertools
import dill as pickle
import string
from collections import Counter

# NLP tools
# import enchant
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from pymystem3 import Mystem
import pymorphy2

import word_lists
# from Analytics.optimization import Dataset

# Init project path
# inspect.getfile(inspect.currentframe()) # script filename (usually with path)
proj_path = '/'.join(inspect.getfile(inspect.currentframe()).split('/')[:-2])
# proj_path = '/'.join(os.path.dirname(os.getcwd()).split('/')[:-1])
print('project path (preprocessing): ', proj_path)

class Preprocessing():
    def __init__(self):
        self.mystem = Mystem()
        self.morph = pymorphy2.MorphAnalyzer()
        # self.en_dict = enchant.DictWithPWL("en_US", proj_path + '/Preprocessing/Dicts/IT_EN_dict.txt')
        # self.ru_aot_dict = enchant.Dict("ru_RU")
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(word_lists.yandex_seo_stopwords)

        self.padding_punctuation = '!"#$%&\'()*+,;<=>?[\\]^`{|}~/«»'
        self.full_punctuation = string.punctuation+'«»'


    def normalize(self, input_string):
        return input_string.lower().strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')


    def pad_punctuation(self, input_stirng):
        """ Used to control tokenization """
        normal_text = input_stirng.strip()
        for char in self.padding_punctuation:
            normal_text = normal_text.replace(char, ' ' + char + ' ')
        return normal_text


    def tokenize(self, input_string):
        return nltk.word_tokenize(input_string)


    def is_punct(self, token):
        """ True only if all chars are punct """
        result = []
        for c in token:
            if c in self.full_punctuation:
                result.append(True)
            else:
                result.append(False)
        return all(result)

    
    def remove_punct(self, tokenlist):
        return [token for token in tokenlist if not self.is_punct(token)]


    def remove_stopwords(self, tokenized_text):
        return [t for t in tokenized_text if t not in self.stop_words]


    def get_pymorphy_lemma(self, token):
        return self.morph.parse(token)[0].normal_form


    def scan_by_vocab(self, text):
        return [t for t in text if t in self.vocab]


    def apply_pipeline(self, raw_string):
        """ Apply all the methoth one by one for new text """

        normalized = self.normalize(raw_string)
        padded = self.pad_punctuation(normalized)
        tokenized = self.tokenize(padded)
        no_punct = self.remove_punct(tokenized)
        no_stops = self.remove_stopwords(no_punct)
        return no_stops



def print_copies(texts):
    """
    # Check for copies in list
    # Prints number of copies
    """
    length = len(texts)
    length2 = len(list(set(texts)))
    if length != length2:
        common_docs = Counter(texts).most_common()
        # More then one time in corpus
        duplicated = [each for each in common_docs if each[1]>1]
        print(duplicated)
    else:
        print('No copies found\n')


def check_empty_text(texts):
    """
    Check for empty elements in a list of texts
    Print number of empty element
    """
    # similar_ids = [model.docvecs.most_similar(positive=[each], topn=1)[0][0] for each in self.incident_ids]
    for i, text in enumerate(texts):
        if texts[i] == "":
            print(i, 'th document is empty')


def strip_lines(texts):
    """
    Takes raw texts list as input
    Removes whitespace characters in the end and the beginning of each string of each text
    """
    out = []
    for text in texts:
        strings = text.split('\n')
        for string in strings:
            strings[strings.index(string)] = string.strip()
        out.append("\n".join(strings))
    return out


def tokenize(texts):
    """
    Takes raw texts list as input
    Returns tokenized texts list
    """
    out = []
    for text in texts:
        # Tokenizing
        tokens = nltk.word_tokenize(text)
        # Punctuation removing
        tokens = [i for i in tokens if (i not in string.punctuation)]
        """
        # Removing special symbols
        tokens = [i.replace("«", "").replace("»", "") for i in tokens]
        tokens = [i.replace(":", "").replace(":", "").replace(":", "") for i in tokens]
        tokens = [i.replace(".", "").replace(".", "").replace(".", "").replace(".", "") for i in tokens]
        tokens = [i.replace(",", "").replace(",", "").replace(",", "").replace(",", "") for i in tokens]
        """
        out.append(tokens)
    return out


def remove_stopwords(tokenized_texts):
    """
    # Takes tokenized texts list as input
    # Removing stop words and tokens with length < 2
    """
    stop_words = stopwords.words('english')
    stop_words.extend(word_lists.stop_words)

    out = []
    for text in tokenized_texts:
        tokens = []
        for token in text:
            if token not in stop_words and len(token) > 1:
                tokens.append(token)
        out.append(tokens)
    return out


def replace_numbers(tokenized_texts):
    """
    # Takes tokenized texts list as input
    Replacing numeric-containing tokens with special tags
    """
    out = []
    dates = []
    one_num_punct = []
    trash = []
    numbers = []
    for tokens in tokenized_texts:
        for x, token in enumerate(tokens):
            # Separate token by symbol category
            puncts = [symbol for symbol in token if (symbol in string.punctuation)]
            numerics = [symbol for symbol in token if (symbol.isnumeric())]
            alphabetic = [symbol for symbol in token if (symbol.isalpha())]
            # print(alphabetic)

            # If token contains all (trash)
            if (puncts and numerics) and alphabetic:
                one_num_punct.append(token)
                trash.append(token)
                tokens.remove(token)

            # Dates, timestamps, etc
            elif puncts and numerics:
                if len(numerics) == 1:
                    one_num_punct.append(token)
                    tokens[x] = "###CHISLITELNOE###"
                else:
                    dates.append(token)
                    tokens[x] = "###DATE###"

            elif puncts and not numerics and not alphabetic:
                trash.append(token)
                tokens.remove(token)
            elif numerics and not alphabetic and not puncts:
                numbers.append(token)
                tokens[x] = "###NUM###"
        out.append(tokens)
        # print('punct+num:', one_num_punct)
    return out


def make_chain_token_list(tokenized_texts):
    """
    Takes tokenized texts as input
    Makes chain list of  tokens
    """
    return list(itertools.chain.from_iterable(tokenized_texts))


def count_tokens(tokenized_texts):
    """
    Takes tokenized texts as input
    Return count [all, unique] tokens for text list
    """
    return [len(list(itertools.chain.from_iterable(tokenized_texts))),
            len(set(itertools.chain.from_iterable(tokenized_texts)))]


def get_cyrrilic_token_list(tokenized_texts):
    """
    Takes tokenized texts list as input
    Returns list of unique cyrrilic tokens
    """
    out = []
    bad_tokens = []
    for text in tokenized_texts:
        toks = []
        for token in text:
            if len(token) > 1:
                # Cyryllic
                if not is_cyrillic(token) and token.isalpha():
                    if ru_aot_dict.check(token):
                        toks.append(token)
                    else:
                        bad_tokens.append(token)
                else:
                    bad_tokens.append(token)
        out.append(toks)
    return list(set(out))


def get_tokens_analysis(tokens):
    """
    ! Only takes a list of unique tokens as input
    Returns raw analysis
    Which is a list of dicts like:
    {'text': '??????', 'analysis': [{'lex': '??????', 'gr': 'ANUM=(?,??.?,?.?)'}]}
    """
    text = " ".join(tokens)
    analysis = mystem.analyze(text)
    return analysis


def lemmatize_texts(tokenized_texts):
    """
    Takes tokenized texts
    Returns raw analysis
    Which is a list of dicts like:
    {'text': 'text', 'analysis': [{'lex': 'text, 'gr': 'ANUM=(ïð,åä,ìóæ|ïð,åä,ñðåä)'}]}
    """
    flat_tokens = make_chain_token_list(tokenized_texts)
    unique_tokens = list(set(flat_tokens))
    tags = get_tokens_analysis(unique_tokens)

    # Make unique tokens dict {token:lemma}
    token_lemmas = {}
    for tag in tags:
        if tag['text'].isalpha():
            if tag['analysis']:
                token_lemmas[tag['text']] = tag['analysis'][0]['lex']

    tagged_tokens = list(token_lemmas.keys())
    # Change every token to its lemma if it has
    out = []
    for text in tokenized_texts:
        token_list = []
        for token in text:
            if token in tagged_tokens:
                lemma = token_lemmas[token]
                token_list.append(lemma)
            else:
                token_list.append(token)
        out.append(token_list)
    return out


def get_tokens_and_tags(tokenized_texts):
    """
    Takes tokenized texts list as input
    Returns dict of unique tokens with its tags
    """
    # Analysys list for every unique tokens
    analysis = get_tokens_analysis(list(set(make_chain_token_list(tokenized_texts))))

    # Make unique token dict : {token: ['tag', 'tag', 'tag', 'tag']}
    token_and_tags = {}
    for each in analysis:
        # if values non consists of ' ' and '\n', alphabetical and not empty
        if each['text'].isalpha():
            if each['analysis']:
                       # token                                 tags
                token_and_tags[each['text']] = each['analysis'][0]['gr'].split(',')
                # print(each['analysis'][0]['gr'].split(','))
    return token_and_tags

"""
def cut_mails_signature(raw_texts):
    out = []
    for text in raw_texts:
        # if there is at least one '\n'
        deleter = 0
        if '\n' in text:
            strings = text.split('\n')
            for string in strings:
                if deleter:
                    strings[strings.index(string)] = ""
                else:
                    if string.startswith(word_lists.start_phrases):
                        strings[strings.index(string)] = ""
                        deleter = 1
            out.append("\n".join(strings))
        else:
            out.append(text)
    return out
"""


def cut_mails_signature(raw_texts):
    """
    Removes text after words defining signature in the mail
    """
    out = []
    for text in raw_texts:
        # if there is at least one '\n'
        if '\n' in text:
            strings = text.split('\n')
            for each in strings:
                if each.startswith(word_lists.start_phrases):
                    strings[strings.index(each):] = ""
            out.append("\n".join(strings))
        else:
            out.append(text)
    return out



def filter_by_start_words(texts):
    """
    Removes whole string if it starts with symbols listed in word_lists.py
    """
    out = []
    for text in texts:
        if '\n' in text:
            strings = text.split('\n')
            for i, each in enumerate(strings):
                if strings[i].startswith(word_lists.start_tokens):
                    strings[i] = ""

            out.append("\n".join(strings))
        else:
            out.append(text)
    return out


def cut_mails_by_surname(tokenized_texts):
    """
    Removes text after two tokens tagged as [surname] and [name]
    """
    token_analysis = get_tokens_and_tags(tokenized_texts)
    token_list = token_analysis.keys()

    out = []
    for text in tokenized_texts:
        deleter = 0
        out_tokens = []
        for i, token in enumerate(text):
            if not deleter:
                # If token = Cyrillic
                if not is_cyrillic(text[i]):
                    if text[i] in token_list:
                        if 'фам' in token_analysis[text[i]]:
                            if i+1 < len(text) and text[i+1] in token_list:
                                if 'имя' in token_analysis[text[i + 1]]:
                                    deleter = 1
                        elif 'имя' not in token_analysis[text[i]]:
                            out_tokens.append(text[i])
                    else:
                        out_tokens.append(text[i])
                else:
                    out_tokens.append(text[i])
        out.append(out_tokens)
    return out


def remove_tokens_by_tags(tokenized_texts):
    """
    Removes tokens which have tags listed in word_lists.py
    """
    token_analysis = get_tokens_and_tags(tokenized_texts)
    token_list = token_analysis.keys()

    out = []
    bad_tokens = []
    for text in tokenized_texts:
        out_tokens = []
        for i, token in enumerate(text):
            if token in token_list:
                tags = token_analysis[text[i]]
                if any(each in tags for each in word_lists.forbidden_mystem_tags):
                    bad_tokens.append([text[i], tags])
                else:
                    out_tokens.append(text[i])
            else:
                out_tokens.append(text[i])
        out.append(out_tokens)
    # print(list(set(bad_tokens)))
    return out


def is_cyrillic(token):
    """
    Checks if string is of ASCII symbols
    """
    return not(any(ord(c) < 128 for c in token))


def filter_by_dict(tokenized_texts, print_bad=False):
    """
    Removes non-dictionary tokens
    """
    out = []
    bad_tokens = []

    for text in tokenized_texts:
        tokens = []
        for token in text:
            # print(token)
            # Check by dicts
            in_it_ru = True if token in it_ru_dict.it_ru_tokens else False
            in_ru = ru_aot_dict.check(token)
            in_en = en_dict.check(token)

            # print(in_ru, in_aot, in_en)

            if in_ru or in_it_ru or in_en:
                tokens.append(token)
            else:
                bad_tokens.append(token)

        out.append(tokens)
    if print_bad:
        print('bad_tokens', set(bad_tokens))
    return out



def get_frequency_dist(tokenized_texts):
    """
    Takes tokenized texts as input
    Returns frequency distribution of tokens
    """
    freq_distribution = FreqDist(make_chain_token_list(tokenized_texts))
    return freq_distribution


def print_most_common(tokenized_texts, top=50):
    """
    Print TOP most frequent tokens of text corpora
    """
    tops = get_frequency_dist(tokenized_texts).most_common(top)
    print(top, 'Most common tokens: \n')
    for each in tops:
        print(each)


def print_unique_tokens(tokenized_texts):
    """
    Prints hepaxes(unique tokens) of text corpora
    """
    hapaxes = get_frequency_dist(tokenized_texts).hapaxes()
    print('Unique tokens: \n')
    for hapax in hapaxes:
        print(hapax)


def remove_unique_tokens(tokenized_texts):
    """
    Takes tokens of texts as input
    Removes tokens which occur only once in whole corpus
    """
    out = []
    hepaxes = get_frequency_dist(tokenized_texts).hapaxes()
    for text in tokenized_texts:
        tokens = [i for i in text if i not in hepaxes]
        out.append(tokens)
    return out


def save_result(tokenized_texts, save_path):
    with open(save_path, 'wb') as fp:
        pickle.dump(tokenized_texts, fp)
    print('saved')


def preprocessing_line(raw_text):
    """
    Takes raw text as input
    Applies all preprocessing to it
    """
    assert type(raw_text)==str, 'Input raw_text is not string'
    raw_text = raw_text.lower().strip()
    raw_text = [raw_text]
    cut_sig = cut_mails_signature(raw_text)
    filtered = filter_by_start_words(cut_sig)
    tokens = tokenize(filtered)
    # print(tokens)
    cut_mails = cut_mails_by_surname(tokens)
    # print(cut_mails)
    lemmatized = lemmatize_texts(cut_mails)
    # print(lemmatized)
    no_stop = remove_stopwords(lemmatized)
    # print(no_stop)
    by_dict = filter_by_dict(no_stop)
    # print(by_dict)
    by_tags = remove_tokens_by_tags(by_dict)
    # print(by_tags)
    preprocessed_text = replace_numbers(by_tags)
    # print('result', preprocessed_text)
    # unpack listed result
    return preprocessed_text

