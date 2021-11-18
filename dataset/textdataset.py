import torch
from language.corpus import DualingualCorpus

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