import re
import nltk
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import RSLPStemmer
import enchant
import unidecode

class NLPreprocessing():
    """NPL preprocessing tasks
    """
    def __init__(self, stopwords_lang='portuguese', spelling_lang='pt_BR') -> None:
        # download list of stopwords
        nltk.download('stopwords')
        # RSLPStemmer
        nltk.download('rslp')
        # WordNetLemmatizer
        nltk.download('wordnet')

        self.stopwd = stopwords.words(stopwords_lang)
        self.ent = enchant.Dict(spelling_lang)

    def remove_punctuations(self, text:str) -> str:
        """
            Remove all ponctuations in the text input
        """
        punct = string.punctuation

        return text.translate(str.maketrans('', '', punct))

    def remove_stopwords(self, text:str) -> str:
        """
            Revemo all stopwords of a text input
        """
        return " ".join([word for word in text.split() if word not in self.stopwd])

    def frequent_words(self, list_words:list, num_top_frequency:int = 10) -> list:
        """
            Return a list of words frequency
        """
        freq = Counter()

        for word in list_words:
            # counting each word
            freq[word] += 1

        return freq.most_common(num_top_frequency)

    def remove_unknown_words(self, text:str) -> str:
        """
            Removing unknown words <= 2 after removing stopwords
        """
        return " ".join([word for word in text.split() if len(word) > 2])

    def remove_freq_words(self, text:str, top_frequency_word:int = 10) -> str:
        """
            Remove all stopwords in the input text
        """

        freq_words = [word for (word, _) in NLPreprocessing.frequent_words(text=text, num_top_frequency=top_frequency_word)]

        return " ".join([word for word in text.split() if word not in freq_words])

    def remove_rare_words(self, text:str) -> str:
        """
            List of rare words
        """

        freq_words = [word for (word, freq) in NLPreprocessing.frequent_words(text=text, num_top_frequency=None) if freq == 1]

        # reversing top freq words to get all the rare one.
        rare_words = freq_words[::-1]
        
        return " ".join([word for word in text.split() if word not in rare_words])

    def steamming(self, text:str) -> str:
        """
            Reduces eatch word to the root
        """
        steammer = RSLPStemmer()

        return " ".join([steammer.stem(word) for word in text.split()])

    def remove_urls(self, text:str) -> str:
        """
            Remove all url symbols from the input text
        """
        upattern = re.compile(r'https?://\S+|www\.\S+')
        return upattern.sub(r'', text)

    def remove_tags(self, text):
        """
            Remove all tags from the input text
        """
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)
    
    def remove_misspelled_word(self, text):
        """Remove misspelled word from a input text
        """
        return " ".join([word.strip() for word in text.split(' ') if self.spelling_check(word.strip())])
    
    def spelling_check(self, word, lang='pt_BR'):
        """Check whether an input word is spelled correctly.
        This preprocessing step needs the input text has alread
        done some nlp preprocessing, kind of: ponctuations and 
        accents removal.
        """
        
        try:
            if not self.ent.check(word):
                suggest = self.ent.suggest(word)
                for sug in suggest:
                    sug = self.remove_accents(sug)
                    if word == sug:
                        return True
                return False
            return True
        except:
            # Its a space char
            return False
    
    def remove_accents(self, text):
        """Remove accents for an input text
        """
        return unidecode.unidecode(text)
    
    def remove_digits(self, text:str):
        """Remove all digits from an input text
        """
        return re.sub("\d+", "", text)
    
    def filter_text(self, text:str):
        text = self.remove_punctuations(text.lower())
        text = self.remove_digits(text)
        text = self.remove_accents(text)
        text = self.remove_misspelled_word(text)
        text = self.remove_urls(text)
        text = self.remove_tags(text)
        text = self.remove_stopwords(text)
        text = self.remove_unknown_words(text)
        text = self.steamming(text)
        return text