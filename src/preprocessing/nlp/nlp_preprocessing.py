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
        Remove punctuation characters from the input text and return the cleaned text.

        Parameters:
        text (str): The input text from which to remove punctuation characters.

        Returns:
        str: The cleaned text with punctuation characters removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with punctuation
        cleaned_text = self.remove_punctuations(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes all punctuation characters (e.g., symbols, commas, periods) from it,
        returning a cleaned version of the text with punctuation removed.
        """
        punct = string.punctuation

        return text.translate(str.maketrans('', '', punct))

    def remove_stopwords(self, text:str) -> str:
        """
        Remove common stopwords from the input text and return the cleaned text.

        Parameters:
        text (str): The input text from which to remove stopwords.

        Returns:
        str: The cleaned text with stopwords removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with stopwords
        cleaned_text = self.remove_stopwords(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes common stopwords (e.g., 'the', 'and', 'in') from it, returning
        a cleaned version of the text with stopwords removed.
        """
        return " ".join([word for word in text.split() if word not in self.stopwd])

    def frequent_words(self, list_words:list, num_top_frequency:int = 10) -> list:
        """
        Find the most frequently occurring words in a list of words.

        Parameters:
        list_words (list): A list of words to analyze for frequency.
        num_top_frequency (int, optional): The number of top frequent words to retrieve. Default is 10.

        Returns:
        list: A list of tuples containing the top 'num_top_frequency' frequent words and their respective frequencies.

        Example:
        ```
        # Assuming 'word_list' is a list of words
        top_frequent_words = self.frequent_words(word_list, num_top_frequency=10)
        print(top_frequent_words)
        ```
        This method analyzes a list of words and returns a list of the most frequently occurring words along with their frequencies.
        You can specify the number of top frequent words to retrieve using the 'num_top_frequency' parameter.
        """
        freq = Counter()

        for word in list_words:
            # counting each word
            freq[word] += 1

        return freq.most_common(num_top_frequency)

    def remove_unknown_words(self, text:str) -> str:
        """
        Remove words from the input text that are shorter than a specified length.

        Parameters:
        text (str): The input text from which to remove short words.
        
        Returns:
        str: The cleaned text with short words removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with short words
        cleaned_text = self.remove_unknown_words(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes words that are shorter than a specified length (default is 2 characters).
        It returns a cleaned version of the text with short words removed.
        """
        return " ".join([word for word in text.split() if len(word) > 2])

    def remove_freq_words(self, text:str, top_frequency_word:int = 10) -> str:
        """
        Remove the most frequently occurring words from the input text.

        Parameters:
        text (str): The input text from which to remove frequent words.
        top_frequency_word (int, optional): The number of top frequent words to remove. Default is 10.

        Returns:
        str: The cleaned text with the top frequent words removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with frequent words
        cleaned_text = self.remove_freq_words(text, top_frequency_word=10)
        print(cleaned_text)
        ```
        This method takes an input text and removes the top 'top_frequency_word' most frequently occurring words from it.
        It returns a cleaned version of the text with the frequent words removed.
        """
        freq_words = [word for (word, _) in NLPreprocessing.frequent_words(text=text, num_top_frequency=top_frequency_word)]

        return " ".join([word for word in text.split() if word not in freq_words])

    def remove_rare_words(self, text:str) -> str:
        """
        Remove rare words (words that occur only once) from the input text.

        Parameters:
        text (str): The input text from which to remove rare words.

        Returns:
        str: The cleaned text with rare words removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with rare words
        cleaned_text = self.remove_rare_words(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes words that occur only once in the text, which are considered rare words.
        It returns a cleaned version of the text with rare words removed.
        """
        freq_words = [word for (word, freq) in NLPreprocessing.frequent_words(text=text, num_top_frequency=None) if freq == 1]

        # reversing top freq words to get all the rare one.
        rare_words = freq_words[::-1]
        
        return " ".join([word for word in text.split() if word not in rare_words])

    def steamming(self, text:str) -> str:
        """
        Apply stemming to the words in the input text to reduce them to their root forms.

        Parameters:
        text (str): The input text to which stemming will be applied.

        Returns:
        str: The text with words reduced to their root forms using stemming.

        Example:
        ```
        # Assuming 'text' is a string containing words to be stemmed
        stemmed_text = self.stemming(text)
        print(stemmed_text)
        ```
        This method takes an input text and applies stemming to reduce words to their root forms. It is useful for
        normalizing words in text data.
        """
        steammer = RSLPStemmer()

        return " ".join([steammer.stem(word) for word in text.split()])

    def remove_urls(self, text:str) -> str:
        """
        Remove URLs (web links) from the input text.

        Parameters:
        text (str): The input text from which to remove URLs.

        Returns:
        str: The text with URLs removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with URLs
        cleaned_text = self.remove_urls(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes URLs, including both 'http://...' and 'www...' patterns, to clean
        the text from web links.
        """
        upattern = re.compile(r'https?://\S+|www\.\S+')
        return upattern.sub(r'', text)

    def remove_tags(self, text):
        """
        Remove HTML tags from the input text.

        Parameters:
        text (str): The input text from which to remove HTML tags.

        Returns:
        str: The text with HTML tags removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with HTML tags
        cleaned_text = self.remove_tags(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes any HTML tags (e.g., <div>, <p>, <a>) from it, returning a
        cleaned version of the text with the tags removed.
        """
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)
    
    def remove_misspelled_word(self, text):
        """
        Remove misspelled words from the input text based on a spelling check.

        Parameters:
        text (str): The input text from which to remove misspelled words.

        Returns:
        str: The text with misspelled words removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with misspelled words
        cleaned_text = self.remove_misspelled_word(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes words that are identified as misspelled based on a spelling check.
        It returns a cleaned version of the text with misspelled words removed.
        """
        return " ".join([word.strip() for word in text.split(' ') if self.spelling_check(word.strip())])
    
    def spelling_check(self, word, lang='pt_BR'):
        """
        Check the spelling of a word in the specified language and return True if the word is correctly spelled.

        Parameters:
        word (str): The word to check for correct spelling.
        lang (str, optional): The language for which to perform the spelling check. Default is 'pt_BR' (Brazilian Portuguese).

        Returns:
        bool: True if the word is correctly spelled, False otherwise.

        Example:
        ```
        # Assuming 'word' is a word to be checked for correct spelling
        is_spelled_correctly = self.spelling_check(word)
        if is_spelled_correctly:
            print(f"'{word}' is spelled correctly.")
        else:
            print(f"'{word}' is misspelled.")
        ```
        This method checks the spelling of a word in the specified language (default is Brazilian Portuguese) using
        language-specific spelling dictionaries. It returns True if the word is spelled correctly, and False if it's misspelled.
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
        """
        Remove diacritics (accents) from characters in the input text.

        Parameters:
        text (str): The input text from which to remove diacritics.

        Returns:
        str: The text with diacritics (accents) removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with diacritics
        cleaned_text = self.remove_accents(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes diacritics (accents) from characters, providing a version of the text
        with diacritics removed. It's useful for text normalization and accent-insensitive comparisons.
        """
        return unidecode.unidecode(text)
    
    def remove_digits(self, text:str):
        """
        Remove digits (numeric characters) from the input text.

        Parameters:
        text (str): The input text from which to remove digits.

        Returns:
        str: The text with digits (numeric characters) removed.

        Example:
        ```
        # Assuming 'text' is a string containing text with digits
        cleaned_text = self.remove_digits(text)
        print(cleaned_text)
        ```
        This method takes an input text and removes all numeric characters (digits) from it, providing a version of the text
        with digits removed. It's useful for text processing and analysis that doesn't require numeric information.
        """
        return re.sub("\d+", "", text)
    
    def filter_text(self, text:str):
        """
        Apply a series of text processing steps to clean and filter the input text.

        Parameters:
        text (str): The input text to be processed and filtered.

        Returns:
        str: The processed and filtered text.

        Example:
        ```
        # Assuming 'text' is a string containing text to be filtered
        filtered_text = self.filter_text(text)
        print(filtered_text)
        ```
        This method applies a series of text processing steps to clean and filter the input text. It includes removing
        punctuation, digits, accents, misspelled words, URLs, HTML tags, stopwords, unknown words, and performing stemming.
        The result is a cleaned and filtered version of the input text.
        """
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