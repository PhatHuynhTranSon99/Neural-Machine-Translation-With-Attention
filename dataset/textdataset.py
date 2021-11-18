from typing import *
import torch
import numpy as np
from language.corpus import DualingualCorpus

def max_length(sentences: List[List[str]]) -> int:
    """
    Find the length of the longest sentence in a sentence list

    Params:
    sentences (List[List[str]]): list of sentences (which are splitted into tokens)

    Returns:
    the length of the longest sentences
    """
    return max([len(sentence) for sentence in sentences])


def pad_sentences(sentences: List[List[str]], pad_index=1) -> List[List[int]]:
    """
    Pad sentences so that all sentences have same the same length, which is the length of the
    longest sentence.

    Params:
    sentences (List[List[str]]): list of sentences (which are splitted into tokens)
    pad_index (int): The index of pad token. Default to 1

    Returns: 
    List of all padded sentences where all sentences have the same length
    """
    # Find the longest sentence's length
    max_len = max_length(sentences)

    # Pad every sentence to max_len
    padded_sentences = []
    for sentence in sentences:
        # Pad sentence to get padded_sentence
        if len(sentence) < max_len:
            # Pad the end of tokens with pad_index
            padded_sentence = sentence + [pad_index] * (max_len - len(sentence)) 
        else:
            padded_sentence = sentence

        # Append to padded_sentences
        padded_sentences.append(padded_sentence)

    return padded_sentences


def collate_fn(batch: List[Tuple[List[int], List[int], int]]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Preprocess the batch so that source_sentences are sorted by length descending
    Params: batch is a list containing batch_size number of samples

    Returns: Tuple(source_sentences, target_sentences, source_length)
    source_sentences (torch.Tensor): batch of padded source sentences of size (max_source_sentences_length, batch_size)
    target_sentences (torch.Tensor): batch of padded source sentences of size (max_target_sentences_length, batch_size)
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

    # Add padding to source_sentences and target_sentences
    source_sentences = pad_sentences(source_sentences)
    target_sentences = pad_sentences(target_sentences)

    # Convert source_sentences and target_sentences to tensor
    # source_sentence_as_tensor has shape (batch_size, max_source_len)
    # target_sentence_as_tensor has shape (batch_size, max_target_len)
    source_sentences_as_tensor = torch.tensor(source_sentences, dtype=torch.long)
    target_sentences_as_tensor = torch.tensor(target_sentences, dtype=torch.long)

    # Transpose source_sentences_as_tensor to (max_source_len, batch_size)
    # target_sentences_as_tensor to (max_source_len, batch_size)
    source_sentences_as_tensor = source_sentences_as_tensor.transpose(1, 0)
    target_sentences_as_tensor = target_sentences_as_tensor.transpose(1, 0)

    return source_sentences_as_tensor, target_sentences_as_tensor, source_lengths


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