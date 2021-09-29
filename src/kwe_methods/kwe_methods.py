#packages for individual KWE methods
import yake
from rake_nltk import Rake

# further internal packages
from util.utilities import safe_division, convert_3_grams_into_single_keywords
from .kwe_base_method import BaseKeywordextractor

# further external packages
import pandas as pd

# --- BEGIN YAKE KWE ---
class YakeKeywordextractor(BaseKeywordextractor):
    """Class used for the yake (Yet Another Keyword Extractor) method

    Class Variables
    -------
    document: str (inherited from BaseKeywordextractor)
        text that is supposed to be used for deriving the Keywords

    sig_size: int (inherited from BaseKeywordextractor)
        count of the top k keywords that should be returned from the KWE method

    n_gram_size: int
        maximum size of the n_grams that are supposed to be returned from the KWE method

    Notes
    -----
    This class represents the yake KWE method. Source: https://github.com/LIAAD/yake
    weights are returned normalized and inverted because in yake a small score means that a keyword is relevant
    """

    def __init__(self, document, sig_size, n_gram_size=1):
        super().__init__(document, sig_size)
        self.n_gram_size = n_gram_size

    def run_extraction(self):
        """implementation of the extraction process of the yake method.

           Returns (inherited from BaseKeywordextractor.run_extraction())
           -------
           top_k_keywords : list of str
              list of the top k keywords from the document
           normalized_word_weights : list of floats
              normalized word weights of the top k keywords
        """

        # calls the yake init from the yake package
        y = yake.KeywordExtractor(n=self.n_gram_size, top=self.sig_size)

        # extracts the keywords and their respective relative weight as two separate lists
        top_k_keywords, word_weights = map(list, zip(*y.extract_keywords(self.document))) # CANNOT HANDLE EMPTY self.document

        # invert the weights --> (high = important)
        max_weight = max(word_weights)
        word_weights_inverted = [max_weight - weight_importance for weight_importance in word_weights]

        # normalizes the weights such that they sum to 1
        sum_of_word_weights = sum(word_weights_inverted)
        normalized_word_weights = [safe_division(el, sum_of_word_weights) for el in word_weights_inverted]

        # if n_grams of size greater than 1 were allowed for the extraction process, the top 20 individual keyWORDS have to be used for the further pipeline process
        if self.n_gram_size > 1:
            top_k_keywords, word_weights = convert_3_grams_into_single_keywords(self.sig_size, top_k_keywords, normalized_word_weights)

        return top_k_keywords, normalized_word_weights
# --- END YAKE KWE ---


# --- BEGIN RAKE KWE ---
class RakeKeywordextractor(BaseKeywordextractor):
    """Class used for the RAKE (Rapid Automatic Keyword Extraction ) method

     Class Variables
     -------
     document: str (inherited from BaseKeywordextractor)
         text that is supposed to be used for deriving the Keywords

     sig_size: int (inherited from BaseKeywordextractor)
         count of the top k keywords that should be returned from the KWE method

     n_gram_size: int
         maximum size of the n_grams that are supposed to be returned from the KWE method

     Notes
     -----
     This class represents the RAKE method. Source: https://pypi.org/project/rake-nltk/
     weights are returned normalized.
     """

    def __init__(self, document, sig_size, n_gram_size=1):
        super().__init__(document, sig_size)
        self.n_gram_size = n_gram_size

    def run_extraction(self):
        """implementation of the extraction process of the RAKE method.

        Returns (inherited from BaseKeywordextractor.run_extraction())
        -------
        top_k_keywords : list of str
            list of the top k keywords from the document
        normalized_word_weights : list of floats
            normalized word weights of the top k keywords
        """

        # initializes a Rake object
        r = Rake(max_length=self.n_gram_size)

        # removes any character that is not a letter, number or a punctuation mark
        nospecialcharacter = True
        if (nospecialcharacter):
            temp_document = ""
            for letter in self.document:
                if str.isalnum(letter) or letter == '.' or letter == ' ' or letter == '!' or letter == '?':
                    temp_document = temp_document + letter
            self.document = temp_document

        # extracts the keywords from the document using the RAKE KWE method
        r.extract_keywords_from_text(self.document)

        # gets the keywords with their respective scores and then splits the dictionary in two lists
        word_weights, keywords = map(list, zip(*r.get_ranked_phrases_with_scores()))

        # slices the lists so that only the top k keywords and their scores get returned
        top_k_word_weights = word_weights[:self.sig_size]
        top_k_keywords = keywords[:self.sig_size]

        # normalizes the weights such that they sum to 1
        sum_of_word_weights = sum(top_k_word_weights)
        normalized_word_weights = [safe_division(el, sum_of_word_weights) for el in top_k_word_weights]

        # if n_grams of size greater than 1 were allowed for the extraction process, the top 20 individual keyWORDS have to be used for the further pipeline process
        if self.n_gram_size > 1:
            top_k_keywords, normalized_word_weights = convert_3_grams_into_single_keywords(self.sig_size, top_k_keywords, normalized_word_weights)

        return top_k_keywords, normalized_word_weights
# --- END RAKE KWE ---

# --- BEGIN TFIDF KWE ---
class TfidfKeywordextractor(BaseKeywordextractor):
    """Class used for the tfidf (term-frequency inverse-document-frequency) KWE method

    Class Variables
    -------
    document: str (inherited from BaseKeywordextractor)
        text that is supposed to be used for deriving the Keywords

    sig_size: int (inherited from BaseKeywordextractor)
        count of the top k keywords that should be returned from the KWE method

   tfidf_vectorizer: sklearn.feature_extraction.text.TfidfVectorizer (Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
        Object that was initialized in the pipeline.py because in order to compute the df-values all documents have to be loaded at the same time which is
        not supposed to happen inside of the KeywordExtractor classes of this implementation.
        The object is needed in order to compute the tfidf-scores for the current document of the current instance.

   n_gram_size: int
        maximum size of the n_grams that are supposed to be returned from the KWE method

    Notes
    -----
    This class represents the tfidf KWE method. Source: http://www.tfidf.com/
    Weights are returned normalized
    """
    def __init__(self, document, sig_size, tfidf_vectorizer, n_gram_size=1):
        super().__init__(document, sig_size)
        self.tfidf_vectorizer = tfidf_vectorizer
        self.n_gram_size = n_gram_size

    def run_extraction(self):
        """implementation of the extraction process of the tfidf KWE method.

        Returns (inherited from BaseKeywordextractor.run_extraction())
        -------
        top_k_keywords : list of str
            list of the top k keywords from the document
        normalized_word_weights : list of floats
            normalized word weights of the top k keywords
        """

        # seems weird, but the tfidf_vectorizer.transform method works best when a list is used as input, hence the one-item-list
        docs = []
        docs.append(self.document)

        # takes every unique word from the document and computes its tfidf value by using the idf from the tfidf_vectorizer instance
        tf_idf_vector = self.tfidf_vectorizer.transform(docs)

        # takes the sparse tf_idf_vector and puts it into a pandas dataframe with the terms as index and the tfidf values as one column
        df = pd.DataFrame(tf_idf_vector.T.todense(), index=self.tfidf_vectorizer.get_feature_names(), columns=["tfidf_values"])

        # sorts the dataframe by highest tfidf value and then extracts only the top k rows
        df = df.sort_values(by=["tfidf_values"], ascending=False).head(self.sig_size)

        # splits the dataframe into two lists for the keywords and the weights
        top_k_keywords = list(df.index)
        top_k_word_weights = list(df['tfidf_values'])

        # normalizes the weights such that they sum to 1
        sum_of_word_weights = sum(top_k_word_weights)
        normalized_word_weights = [safe_division(el, sum_of_word_weights) for el in top_k_word_weights]

        # if n_grams of size greater than 1 were allowed for the extraction process, the top 20 individual keyWORDS have to be used for the further pipeline process
        if self.n_gram_size > 1:
            top_k_keywords, normalized_word_weights = convert_3_grams_into_single_keywords(self.sig_size, top_k_keywords, normalized_word_weights)

        return top_k_keywords, normalized_word_weights
# --- END TFIDF KWE ---
