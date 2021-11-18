from typing import *
import torch
import numpy as np
from language.corpus import DualingualCorpus


def collate_fn(batch: List[Tuple[List[int], List[int], int]]):
    """
    Preprocess the batch so that source_sentences are sorted by length descending
    Params: batch is a list containing batch_size number of samples

    Returns: Tuple(source_sentences, target_sentences, source_length)
    source_sentences (List[List[int]]): list of tokenize source sentences
    target_sentences (List[List[int]]): list of tokenize target sentences
    source_length (List[int]): list of lengths of source sentences
    """
    # Extract the source_sentences, target_sentences and source_lengths
    source_sentences = [item[0] for item in batch]
    target_sentences = [item[1] for item in batch]
    source_lengths = [item[2] for item in batch]

    # Sort the index of source_lengths in descending order
    indices_sorted_by_length = list(np.argsort(source_lengths))[::-1]

    # Sort the source sentences, target sentences and source lengths based on the indices
    source_sentences = [source_sentences[index] for index in indices_sorted_by_length]
    target_sentences = [target_sentences[index] for index in indices_sorted_by_length]
    source_lengths = [source_lengths[index] for index in indices_sorted_by_length]

    return source_sentences, target_sentences, source_lengths


"""
Create a custom dualingual dataset
"""
class CustomDualingualDataset(torch.utils.data.Dataset):
    def __init__(self, corpus: DualingualCorpus):
        """
        Params: corpus( DualingualCorpus): the corpus containing dualingual data
        Returns: None
        """
        # Cache the corpus
        self.corpus = corpus

    def __len__(self):
        """
        Return the length of the corpus
        """
        return len(self.corpus.source_sentences_as_index)

    def __getitem__(self, index: int):
        """
        Get a training sample given an index

        Params: index (int) of the training sample
        Returns: the training sample at that index containing 
        (
            source_sentence: List[int]
            target_sentence: List[int],
            length: length of source sentence
        )
        """
        return self.corpus.source_sentences_as_index[index], self.corpus.target_sentences_as_index[index], len(self.corpus.source_sentences_as_index[index]) 