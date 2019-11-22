from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from selector import TextSelector, NumberSelector

from nltk.corpus import stopwords

STOPWORDS = list(set(stopwords.words('english')))


class FeaturePipeline():
    def __init__(self):
        self._create_pipelines()

    def _create_pipelines(self):
        """
        This method creates the pipeline for the different keys of the
        dataframe and initialize the feature extractor constructors for
        numerical and text values.
        """
        self.p_blog = Pipeline([
                ('selector', TextSelector(key='blog')),
                ('tfidf', TfidfVectorizer(stop_words=STOPWORDS))])
                #('normalizer', Normalizer()),
                #('tsvd', TruncatedSVD(n_components=32, random_state=42))])
                
        #self.p_tokens = Pipeline([
        #        ('selector', TextSelector(key='tokens')),
        #        ('tfidf', TfidfVectorizer())])
        #self.p_text_norm = Pipeline([
        #        ('selector', TextSelector(key='text_norm')),
        #        ('tfidf', TfidfVectorizer(stop_words=STOPWORDS))])
        self.p_length =  Pipeline([
                ('selector', NumberSelector(key='length')),
                ('standard', MinMaxScaler())])
        #self.p_word_count =  Pipeline([
        #        ('selector', NumberSelector(key='word_count')),
        #        ('standard', MinMaxScaler())])
        self.p_sentence_count =  Pipeline([
                ('selector', NumberSelector(key='sentence_count')),
                ('standard', MinMaxScaler())])
        #self.p_flesch_reading_ease =  Pipeline([
        #        ('selector', NumberSelector(key='flesch_reading_ease')),
        #        ('standard', MinMaxScaler())])
        #self.p_flesch_kincaid_grade =  Pipeline([
        #        ('selector', NumberSelector(key='flesch_kincaid_grade')),
        #        ('standard', MinMaxScaler())])
        #self.p_difficult_words =  Pipeline([
        #        ('selector', NumberSelector(key='difficult_words')),
        #        ('standard', MinMaxScaler())])
        self.p_bad_word_count =  Pipeline([
                ('selector', NumberSelector(key='bad_word_count')),
                ('standard', MinMaxScaler())])
        #self.p_pos_tag =  Pipeline([
        #        ('selector', TextSelector(key='pos_tag')),
        #        ('tfidf', TfidfVectorizer())])
    

    def _fit_transform(self, train_X):
        """
        This method fit all the transforms and transform the data.
        
        Parameter:
                train_X:  train dataframe
                type = pd.DataFrame
        """
        self.p_blog.fit_transform(train_X)
        #self.p_tokens.fit_transform(train_X)
        #self.p_text_norm.fit_transform(train_X)
        self.p_length.fit_transform(train_X)
        #self.p_word_count.fit_transform(train_X)
        self.p_sentence_count.fit_transform(train_X)
        #self.p_flesch_reading_ease.fit_transform(train_X)
        #self.p_flesch_kincaid_grade.fit_transform(train_X)
        #self.p_difficult_words.fit_transform(train_X)
        self.p_bad_word_count.fit_transform(train_X)
        #self.p_pos_tag.fit_transform(train_X)

    def fit_transform(self, train_X):
        """
        This method concatenates the result of multiple transformer
        objects by extracting all the features into a single transformer.

        Parameter:
                train_X:  train dataframe
                type = pd.DataFrame
        Returns the the pipeline fitted on all of the features results concactenated
                type = sklearn.pipeline.Pipeline
        """
        self._fit_transform(train_X)
        all_features = FeatureUnion([('blog', self.p_blog),
                                    #('tokens', self.p_tokens), 
                                    #('text_norm', self.p_text_norm),
                                    ('length', self.p_length),
                                    #('word_count', self.p_word_count),
                                    ('sentence_count', self.p_sentence_count),
                                    #('flesch_reading_ease', self.p_flesch_reading_ease),
                                    #('flesch_kincaid_grade', self.p_flesch_kincaid_grade),
                                    #('difficult_words', self.p_difficult_words),
                                    ('bad_word_count', self.p_bad_word_count),
                                    #('pos_tag', self.p_pos_tag)
                                    ])
        p_features = Pipeline([('all_features', all_features)])
        p_features.fit_transform(train_X)
        return p_features