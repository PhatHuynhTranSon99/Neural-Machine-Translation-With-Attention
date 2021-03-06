import torch
from typing import *
from model.encoder import Encoder
from model.decoder import Decoder


class SequenceToSequenceModel(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        """
        Create the sequence to sequence model by combing encoder and decoder

        Params:
        encoder (Encoder): The encoder part
        decoder (Decoder): The decoder part
        """
        # Call parent constructor to initialize model
        super(SequenceToSequenceModel, self).__init__()

        # Save the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, source_sentences: torch.Tensor, target_sentences: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        Perform the forward pass of the model by piping inputs encoder and then decoder

        Params:
        source_sentences (torch.Tensor): Tensor of size (max_source_length, batch_size) containing batch of sentences from source language
        target_sentences (torch.Tensor): Tensor of size (max_target_length, batch_size) containing batch of sentences from target language
        source_lengths (List[int]): List of size (batch_size,) containing the length of each sentence in a batch

        Returns:
        P (torch.Tensor): Tensor of size (max_target_length - 1, batch_size, vocab_size) containing the input of future softmax to calculate 
        probablity of next words
        """
        # pass source_sentences and source_lengths through encoder
        # to get enc_hidden, dec_init_state
        # enc_hidden has size (max_source_sentence_length, batch_size, hidden_size)
        enc_hidden, dec_init_state = self.encoder(source_sentences, source_lengths)

        # Create enc_mask which is a tensor of size (batch_size, max_source_sentence_length)
        # entries with padding will have 1's, with non-padding items will have 0's
        # enc_mask now has size (max_source_sentence_length, batch_size)
        enc_mask = (source_sentences == self.encoder.vocab.get_index_from_word("<pad>")).float()
        # transpose enc_mask to get correct dimension
        # enc_mask now has size (batch_size, max_source_sentence_length)
        enc_mask = enc_mask.transpose(1, 0)

        # Pass padded target sentences, enc_hidden, dec_init_state and enc_mask through decoder to get
        # P: size (max_target_sentence_length - 1, batch_size, vocab_size)
        P = self.decoder(target_sentences, enc_hidden, dec_init_state, enc_mask)

        # Use log_softmax on P to calculate the score on final dimension
        # log_softmax has size (max_target_sentence_length - 1, batch_size, vocab_size)
        log_softmax = torch.nn.functional.log_softmax(P, dim=2)

        # Use negative sign to convert to negative log softmax
        # neg_log_softmax has size (max_target_sentence_length - 1, batch_size, vocab_size)
        neg_log_softmax = - log_softmax

        # Gather value along vocab_size dimension (dim = 2)
        # to get the score for each words

        # Target_indices has size (max_target_length - 1, batch_size)
        # contain the sentences where first words are removed
        target_indices = target_sentences[1:]

        # Unsqueeze to get (max_target_length - 1, batch_size, 1)
        target_indices = target_indices.unsqueeze(2)

        # Use target indices are indices for gathering loss at dimenion of vocab_size
        # losses has size (max_target_length - 1, batch_size, 1)
        losses = torch.gather(neg_log_softmax, index=target_indices, dim=2)

        # squeeze losses to get losses (max_target_length - 1, batch_size)
        losses = losses.squeeze(2)

        # these losses include <pad> token loss -> We need to remove that using a mask
        # target_mask has size (max_sentence_length - 1, batch_size) (without first words)
        # is one where there is a non-padding token, is zero where there is a padding token
        target_mask = (target_sentences != self.decoder.vocab.get_index_from_word("<pad>")).float()
        target_mask = target_mask[1:]

        # Multiply losses with target_mask to get losses in non-padding positions only
        # losses has size (max_target_sentence_length - 1, batch_size)
        losses = losses * target_mask

        # sum of the losses for each sentence to get losses (batch_size,)
        # containing losses for each sentence in the batch
        losses = losses.sum(dim=0)

        return losses