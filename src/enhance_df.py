import pandas as pd
from tqdm.auto import tqdm
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, pos_tag_sents

from tokenizer import Tokenizer
from utils import read_word_list

BAD_WORD_LIST = 'badwords.txt'

readability_stats = [
    #('flesch_reading_ease', 'Flesch Reading Ease'),
    #('flesch_kincaid_grade', 'Flesch-Kincaid Grade Level'), 
    ('difficult_words', 'Difficult Words')]

def enhance_tokenization(df):
    """
    This function enhances the dataframe with the length of blogs,
    the words tokenized, the sentences tokenized, tokens processed by
    our Tokenizer, the word count and the sentence count.

    Parameter:
        df:  dataframe

    Returns the enhanced dataframe
    """
    dataframe = df
    tokenizer = Tokenizer()
    dataframe['length'] = df.blog.str.len()
    tqdm.pandas('Tokenizing Words')
    dataframe['words'] = df.blog.progress_apply(word_tokenize)
    tqdm.pandas('Tokenizing Sentences')
    dataframe['sentences'] = df.blog.progress_apply(sent_tokenize)
    #tqdm.pandas('Tokenizing - Normalized')
    #dataframe['tokens'] = df.blog.progress_apply(tokenizer.process_item)
    #dataframe['tokens'] = dataframe.tokens.progress_apply(lambda x: ' '.join(i for i in x))
    #dataframe['word_count'] = df.words.apply(len)
    dataframe['sentence_count'] = df.sentences.apply(len)
    return dataframe


def enhance_bad_words(df):
    """
    This function enhances the dataframe with the count of
    bad words and a boolean indicating the appearance of bad word.

    Parameter:
        df: dataframe

    Returns the enhanced dataframe
    """
    dataframe = df
    bad_words = read_word_list(BAD_WORD_LIST)
    dataframe['bad_word_count'] = df.words.apply(lambda words: len(set(word.lower() for word in words) & bad_words))
    dataframe['has_bad_words'] = df.bad_word_count > 0
    return dataframe


def enhance_readability(df):
    """
    This function enhances the dataframe with the flesch reading ease,
    the smog index, the flesch kincaid grade, the difficulty of words
    from textstat.

    Parameter:
        df: dataframe

    Returns the enhanced dataframe
    """
    dataframe = df
    for item in readability_stats:
        key, label = item
        tqdm.pandas(desc=label)
        stat = getattr(textstat, key)
        dataframe[key] = df.blog.progress_apply(stat)
    return dataframe


def _join_pos_tag(x):
    """
    This function concatenates the words with their respective tag
    in one string for all pairs in the list.

    Parameter:
        x: a list of pairs (words, tag)

    Returns a string of words with pos tags
    """
    return ' '.join(word[0] + "/" + word[1] for word in x)

def enhance_pos_tag(df):
    """
    This function enhances the dataframe with the part-of-speech
    tagging from nltk and joins the corresponding words with their
    respective tags.

    Parameter:
        df: dataframe

    Returns the enhanced dataframe
    """
    dataframe = df
    dataframe['pos_tag'] = pos_tag_sents(df['tokens'].progress_apply(word_tokenize).tolist())
    dataframe['pos_tag'] = dataframe.pos_tag.apply(_join_pos_tag)
    return dataframe