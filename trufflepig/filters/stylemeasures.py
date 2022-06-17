
import re
import string
from xml.etree.ElementTree import ParseError
import logging

import numpy as np
import scipy.stats as spst
import langdetect
import language_check
import pyphen
from enchant.checker import SpellChecker

logger = logging.getLogger(__name__)

CAPS = "([A-Z])"
PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = "[.](com|net|org|io|gov)"

CONNECTORS = {
    'however',
    'nevertheless',
    'nonetheless',
    'yet',
    'instead',
    'contrary',
    'likewise',
    'similarly',
    'correspondingly',
    'accordingly',
    'consequently',
    'hence',
    'thus',
    'besides',
    'furthermore',
    'moreover',
    'finally',
    'firstly',
    'secondly',
    'thirdly',
    'specifically',
    'instance',
    'indeed',
    'regardimg',
    'conclusion',
    'rather',
    'afterwards',
}

PRONOUNS = {'i', 'me', 'my' , 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'it', 'its', 'we', 'us' 'our', 'they', 'them', 'their'}

STEEMIT_WORDS = (
        'crypto',
        'cryptocurrency',
        'cryptocurrencies',
        'bitcoin',
        'blockchain',
        'litecoin',
        'ethereum',
        'ripple',
        'steem',
        'steemit',
        'resteem',
        'upvote',
        'downvote',
)

def split_into_sentences(text):
    """ Splits a `text` into a list of sentences.

    Taken from https://stackoverflow.com/questions/4576077/python-split-text-on-sentences

    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(PREFIXES, "\\1<prd>", text)
    text = re.sub(WEBSITES, "<prd>\\1", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + CAPS + "[.] ", " \\1<prd> ", text)
    text = re.sub(ACRONYMS + " " + STARTERS, "\\1<stop> \\2", text)
    text = re.sub(CAPS + "[.]" + CAPS + "[.]" + CAPS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(CAPS + "[.]" + CAPS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + SUFFIXES + "[.] " + STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + CAPS + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def count_headings(text):
    return len(re.findall('(\n|^)( *#{1,6}| *<h[1-6]>)[^\n#]*\n', text))


def count_paragraphs(text):
    return text.count('\n\n') + 1


def compute_average_sentence_length(text_list):
    return np.mean([len(x) for x in text_list])


def compute_sentence_length_variance(text_list):
    return np.var([len(x) for x in text_list])


def compute_sentence_length_skew(text_list):
    return spst.skew([len(x) for x in text_list])


def compute_sentence_length_kurtosis(text_list):
    return spst.kurtosis([len(x) for x in text_list])


def compute_average_puncitation(text_list):
    punctuation = [sum(1 for x in sentence if x in set(string.punctuation) )
                         for sentence in text_list]
    return np.mean(punctuation)


def count_characters(tokens):
    return sum(len(x) for x in tokens)


def count_connectors(tokens):
    return sum(1 for x in tokens if x in CONNECTORS)


def count_pronouns(tokens):
    return sum(1 for x in tokens if x in PRONOUNS)
