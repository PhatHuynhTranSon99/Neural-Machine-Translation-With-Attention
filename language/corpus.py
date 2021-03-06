from typing import *
from language.vocab import Vocab

def get_sentences(data_path: str) -> List[str]:
    """
    Read a list of sentences from file
    Params: data_path (str): containing the file path of the corpus
    Returns: a list of sentences in the corpus
    """
    with open(data_path, encoding="utf-8", errors="ignore") as file:
        text = file.read().split("\n")
    return text


def process_sentences(sentences: List[str], is_target: bool = False) -> List[List[str]]:
    """
    Process a list of sentences by removing with space and splitting them
    Params: 
    sentence (List[str]): a list of sentences (list of strings)
    is_target (bool): a boolean denotes if these sentences are from target language

    Returns: a list containing list of tokens , which are splitted sentences
    """
    processed_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        sentence = sentence.split()

        if is_target:
            sentence = ["<sos>"] + sentence + ["<eos>"]

        processed_sentences.append(sentence)

    return processed_sentences


def map_sentences(sentences: List[List[str]], vocab: Vocab) -> List[List[int]]:
    mapped_sentences = []

    for sentence in sentences:
        mapped_sentence = [vocab.get_index_from_word(token) if vocab.contains(token) else vocab.get_index_from_word("<unk>") for token in sentence]
        mapped_sentences.append(mapped_sentence)

    return mapped_sentences


def map_sentences_inverse(sentences: List[List[int]], vocab: Vocab) -> List[List[str]]:
    mapped_sentences = []

    for sentence in sentences:
        mapped_sentence = [vocab.get_word_from_index(index) for index in sentence]
        mapped_sentences.append(mapped_sentence)

    return mapped_sentences


def remove_empty_sentence_pairs(source_sentences: List[List[str]], target_sentences: List[List[str]]) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Remove source and target sentence pairs where the length of the source sentences is 0

    Params: 
    source_sentences (List[List[str]]): list of source sentences (each sentence consists of string tokens)
    target_sentences (List[List[str]]): list of target sentences (each sentence consists off string tokens)

    Returns:
    (source_sentences, target_sentences): sentence pair where empty sentences are removed
    """

    # Find index where source sentences are empty
    source_empty_sentence_indices = []
    for index, sentence in enumerate(source_sentences):
        if len(sentence) == 0:
            source_empty_sentence_indices.append(index)
            print("Removed: ", source_sentences[index], target_sentences[index])

    # Create new source and target sentences
    filtered_source_sentences = []
    filtered_target_sentences = []

    # Remove sentences with index in source_empty_sentences_indices
    for index, sentence in enumerate(source_sentences):
        if index not in source_empty_sentence_indices:
            filtered_source_sentences.append(sentence)

    for index, sentence in enumerate(target_sentences):
        if index not in source_empty_sentence_indices:
            filtered_target_sentences.append(sentence)

    return filtered_source_sentences, filtered_target_sentences

class DualingualCorpus:
    """
    This class is for holding and retrieving dualingual corpus of text (en-vi or vi-en)
    """
    def __init__(self, source_sentences_path: str, target_sentences_path: str):
        """
        Read, process and save sentences from text file

        Params:
        source_sentences_path (str): the file path which holds the text file containing source sentences
        target_sentences_path (str): the file path which holds the text file containing target sentences      

        Returns: None
        """
        # Cache the file path
        self.source_path = source_sentences_path
        self.target_path = target_sentences_path

        # Read the files in and process sentences
        source_sentences = process_sentences(get_sentences(self.source_path))
        target_sentences = process_sentences(get_sentences(self.target_path), is_target=True)

        # There are some sentences where the length is zero -> Remove those sentences from both source and 
        # target sentences
        self.source_sentences, self.target_sentences = remove_empty_sentence_pairs(source_sentences, target_sentences)


    def create_vocabulary(self, corpus_name: str):
        """
        This method is for creating the vocabulary from the sentences

        Params: corpus_name (str): Either 'source' or 'target', denoting the type of corpus to create vocab for
        """
        if corpus_name == "source":
            return Vocab(self.source_sentences)
        elif corpus_name == "target":
            return Vocab(self.target_sentences)


    def convert_words_to_indices(self, source_vocab: Vocab, target_vocab: Vocab):
        """
        Convert sentences from list of words to list of indices using the vocab object

        Params:
        source_vocab (Vocab): The vocab object containing vocabulary of the source sentences
        target_vocab (Vocab): The vocab object containing vocabulary of the target sentences

        Returns: None
        """
        self.source_sentences_as_index = map_sentences(self.source_sentences, source_vocab)
        self.target_sentences_as_index = map_sentences(self.target_sentences, target_vocab)

