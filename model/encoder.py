import torch
from typing import *
from language.vocab import Vocab


class Encoder(torch.nn.Module):
  def __init__(self, vocab: Vocab, embed_size: int, hidden_size: int):
    """
    Params:
    vocab_embedding (VocabEmbedding): Vocab embedding contains embeddings of source language
    embed_size (int): Size of the embeddings
    hidden_size (int): Size of hidden and cell states in LSTM
    
    Returns: None
    """
    # Call super constructor
    super(Encoder, self).__init__()
    
    # Cache embed and hidden size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

    # Cache the vocab
    self.vocab = vocab

    # Save the vocab embedding object as member
    self.vocab_embedding = torch.nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=embed_size,
        padding_idx=vocab.get_index_from_word("<pad>")
    )

    # Create bidirectional LSTM layer
    # with given embedding size as input and hidden_size
    self.encoder_lstm = torch.nn.LSTM(
        embed_size,
        hidden_size,
        bidirectional=True
    )

    # Create the (hidden) h projection linear layer to
    # map enc_final_hidden to dec_init_hidden
    # remember: NO BIAS
    self.h_projection = torch.nn.Linear(
        hidden_size * 2,
        hidden_size,
        bias=False
    )

    # Create the (cell) c projection linear layer to 
    # map enc_final_cell to dec_init_cell
    # remember: NO BIAS
    self.c_projection = torch.nn.Linear(
        hidden_size * 2,
        hidden_size,
        bias=False
    )

  def forward(self, source_padded: torch.Tensor, source_length: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Params:
    source_padded (torch.Tensor): Tensor of size  (max_source_sentence_lengthh, batch_size) containing a batch of source sentences
    source_length (List[int]): A list of length batch_size containing the length of each sentence in the batch

    Returns: (enc_hidden, (dec_init_hidden, dec_init_cell))

    enc_hidden (torch.Tensor): Tensor of size (sentence_length, batch_size, hidden_size * 2) 
    or (sentence_length, batch_size, hidden_size * 2) containing the hidden states of each word of sentence in the batch 
    after going through LSTM. May permute it to get to (batch_size, sentence_length, hidden_size * 2)

    dec_init_hidden (torch.Tensor): Tensor of size (batch_size, hidden_size) as the initial hidden state of decoder LSTM
    dec_init_cell  (torch.Tensor): Tensor of size (batch_size, hidden_size) as the initial cell state of decoder LSTM
    """
    
    # 1) Run source_padded through Embedding to get the tensor X
    # X has size (sentence_length, batch_size, embed_size)
    X = self.vocab_embedding(source_padded)

    # 1.5) Use pack_padded_sequence on X to speed up computation
    X_padded = torch.nn.utils.rnn.pack_padded_sequence(X, source_length)

    # 2) Run X_padded through LSTM to get (enc_hidden, (enc_final_hidden, enc_final_cell))
    # enc_hidden has size (sentence_length, batch_size, hidden_size * 2) (2 because bidirectional)
    # enc_final_hidden has size (2, batch_size, hidden_size) (2 because bidirectional)
    enc_hidden, (enc_final_hidden, enc_final_cell) = self.encoder_lstm(X_padded)

    # 2.5) Convert enc_hidden back to right format using pad_packed_sequence
    # this step is necessary after enc_hidden
    enc_hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_hidden)

    # 2.75) Reshape enc_hidden from (sentence_length, batch_size, hidden_size * 2)
    # to (batch_size, sentence_length, hidden_size * 2)
    enc_hidden = torch.permute(enc_hidden, dims=(1, 0, 2))

    # 3) Concatenate enc_final_hidden to get size (batch_size, hidden_size * 2)
    # by stack them along the second dimension. 
    # Do the same for enc_final_cell
    concatenated_enc_final_hidden = torch.cat((enc_final_hidden[0], enc_final_hidden[1]), dim = 1)
    concatenated_enc_final_cell = torch.cat((enc_final_cell[0], enc_final_cell[1]), dim = 1)

    # 4) Pass concatenated hidden and cell state through linear layer (h_projection and c_projection)
    # to get dec_init_hidden (batch_size, hidden_size) and dec_init_cell (batch_size, hidden_size)
    dec_init_hidden = self.h_projection(concatenated_enc_final_hidden)
    dec_init_cell = self.c_projection(concatenated_enc_final_cell)

    # 5) Return the result
    # enc_hidden: tensor of size (batch_size, sentence_length, hidden_size * 2)
    # (dec_init_hidden, dec_init_cell): both are tensors of size (batch_size, hidden_size)
    return enc_hidden, (dec_init_hidden, dec_init_cell)