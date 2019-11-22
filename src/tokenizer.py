import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

RE_WORD = re.compile(r'^[a-zA-Z]+')
RE_URL = re.compile(r'\w+://\S+')
STOPWORDS = set(stopwords.words('english'))

def filter_no_stopwords(token):
    return token.lower() not in STOPWORDS

def filter_words_only(token):
    return RE_WORD.match(token)

def transform_drop_urls(text):
    return RE_URL.sub('', text)

def transform_lowercase(value):
    return value.lower()

def transform_stem(value):
    return STEMMER.stem(value)

class Tokenizer:
    """
    This class implements standard transformations 
    for tokenization of words.
    """
    DEFAULT_TEXT_TRANSFORMS = [transform_drop_urls]
    DEFAULT_TOKEN_FILTERS = [filter_words_only, filter_no_stopwords]
    DEFAULT_TOKEN_TRANSFORMS = [transform_lowercase]
    def __init__(self):
        self.text_transforms = self.DEFAULT_TEXT_TRANSFORMS
        self.token_filters = self.DEFAULT_TOKEN_FILTERS
        self.token_transforms = self.DEFAULT_TOKEN_TRANSFORMS
        
    def __call__(self, data):
        return (self.process_item(item) for item in show_progress(data, desc='Tokenization'))
    
    def transform(self, token):
        for transform in self.token_transforms:
            token = transform(token)
        return token

    def process_item(self, text):
        """
        This method transform the text by lowering the cases of 
        the document, dropping the urls, create tokens and then
        filtering them to only keep words and no stopwords.

        Parameter:
                text: a document
        Returns a list of the words tokenized
        """
        for text_transform in self.text_transforms:
            text = text_transform(text)
        tokens = word_tokenize(text)
        return [
            self.transform(token) for token in tokens
            if all(
                token_filter(token)
                for token_filter in self.token_filters)]