from typing import *


class Vocab:
    """
    This class holds the vocabulary built from the sentences
    """
    def __init__(self, sentences: List[List[str]], freq_min: int = 5):
        """
        Create vocabulary object from the sentences

        Params: 
        sentences (str): containing the corpus of sentences
        freq_min (int): words with lower than this frequency will be mapped as <unk>
        """
        # First calculate the number of words
        word_count = {}
        for sentence in sentences:
            for word in sentence:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        
        # Convert dictionary to list
        word_count = list(word_count.items())

        # filter item with lower than min frequency
        word_count = filter(lambda x: x[1] >= freq_min, word_count)

        # Get the set of qualified words
        qualified_words = set([item[0] for item in word_count])

        # Now truly find the mapping
        word2index = {
            "<unk>": 0,
            "<pad>": 1,
            "<sos>": 2,
            "<eos>": 3
        }
        current_index = 4

        for sentence in sentences:
            for word in sentence:
                if word in qualified_words and word not in word2index:
                    word2index[word] = current_index
                    current_index += 1

        # Save word2index and index2word as dictionaries
        self.word2index = word2index
        self.index2word = { v: k for k, v in self.word2index.items() }

    
    def __len__(self):
        """
        Returns the number of entries of the vocab
        """
        return len(self.word2index)


    def get_index_from_word(self, word: str) -> int:
        """
        Params: word (str) the word that one needs to find index for
        Returns: the index of that word in the vocab
        """
        return self.word2index[word]


    def get_word_from_index(self, index: int) -> str:
        """
        Params: the index of the word one needs to find
        Returns: the word that has that index in the vocab
        """
        return self.index2word[index]