from typing import *


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
        self.source_sentences = process_sentences(get_sentences(self.source_path))
        self.target_sentences = process_sentences(get_sentences(self.target_path), is_target=True)


    def create_vocabulary(self, corpus_name: str):
        """
        This method is for creating the vocabulary from the sentences

        Params: corpus_name (str): Either 'source' or 'target', denoting the type of corpus to create vocab for
        """
        if corpus_name == "soucre":
            pass
        elif corpus_name == "target":
            pass