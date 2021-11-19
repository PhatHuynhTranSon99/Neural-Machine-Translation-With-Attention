import torch
from typing import *
from language.vocab import Vocab


class Decoder(torch.nn.Module):
  def __init__(self, vocab: Vocab, embed_size: int, hidden_size: int, dropout: float):
    """
    Params:
    vocab_embedding (VocabEmbedding): VocabEmbedding object containing the embeddings of target languages
    embed_size (int): Size of the word embeddings 
    hidden_size (int): Size of hidden and cell states of decoder LSTM (this HAS to be the same with the hidden_size of encoder)

    Returns: None
    """
    # Call parent constructor
    super(Decoder, self).__init__()

    # Cache the embed_size and hidden_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

    # Cache the vocab
    self.vocab = vocab
    
    # Cache the vocab_embedding object as member object
    self.vocab_embedding = torch.nn.Embedding(
        num_embeddings=len(vocab),
        embedding_dim=embed_size,
        padding_idx=vocab.get_index_from_word("<pad>")
    )

    # Create LSTM layer (unidirectional)
    # with embed_size as input_size and hidden_size
    self.decoder_lstm = torch.nn.LSTM(
        embed_size + hidden_size,
        hidden_size,
        bidirectional=False
    )

    # Create attention projection (to calculate the score)
    # remember: NO BIAS
    self.attention_projection = torch.nn.Linear(
        hidden_size * 2,
        hidden_size,
        bias=False
    )

    # Create v projection (to calculate v_t)
    # remember: NO BIAS
    self.v_projection = torch.nn.Linear(
        hidden_size * 3,
        hidden_size,
        bias=False
    )

    # Create target projection (to calculate softmax over target language's tokens)
    # remember: NO BIAS
    self.target_projection = torch.nn.Linear(
        hidden_size,
        len(vocab),
        bias=False
    )

    # Create dropout layer
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, target_padded: torch.Tensor, enc_hidden: torch.Tensor, dec_init_state: Tuple[torch.Tensor, torch.Tensor], enc_mask: torch.Tensor) -> torch.Tensor:
    """
    Params:
    target_padded (torch.Tensor): Tensor of size (max_target_sentence_length, batch_size) containing a batch target sentences

    enc_hidden (torch.Tensor): Tensor of size (batch_size, source_sentence_length, hidden_size * 2) containing computed hidden states of source sentences through encoder LSTM

    dec_init_state (Tuple[torch.Tensor, torch.Tensor]): containing 
    dec_init_hidden (torch.Tensor of size(batch_size, hidden_size))
    dec_init_cell (torch.Tensor of size (batch_size, hidden_size))

    enc_mask (torch.Tensor): Tensor of size (source_sentence_length, batch_size) which is a mask of source sentences.
    entries containing <pad> are 1s, entries containing other tokens are 0s

    Returns:
    P (torch.Tensor): Tensor for size (max_target_sentence_length - 1, batch_size, vocab_size) containing the logists for calculating softmax
    """
    
    # 1) Remove the <end> tokens from sentences whose length is max_target_sentence_length
    # target_padded will have size (max_target_sentence_length - 1, batch_size) after this line
    target_padded = target_padded[:-1]

    # 2) Create Y by passing target_padded through embedding
    # Y will have size (max_target_sentence_length - 1, batch_size, embedding_size)
    Y = self.vocab_embedding(target_padded)

    # 2.5) Create o_prev a zeros tensor of size (batch_size, hidden_size)
    batch_size = Y.shape[1]
    o_prev = torch.zeros(batch_size, self.hidden_size)

    # 2.6) Create combine_outputs as list containing tensors of size (batch_size, vocab_size)
    combined_outputs = []

    # 2.7) Reshape dec_init_hidden and dec_init_cell to right shape
    # dec_init_hidden has size (batch_size, hidden_size) to (1, batch_size, hidden_size)
    # dec_init_hidden has size (batch_size, hidden_size) to (1, batch_size, hidden_size)
    dec_init_hidden, dec_init_cell = dec_init_state
    dec_init_hidden = dec_init_hidden.unsqueeze(0)
    dec_init_cell = dec_init_cell.unsqueeze(0)

    # 2.8) Group into dec_state
    dec_state = (dec_init_hidden, dec_init_cell)

    # 2.9) Pass enc_hidden (batch_size, sentence_length, hidden_size * 2)
    # through attention_projection to get enc_hidden (batch_size, sentence_length, hidden_size)
    enc_hidden_proj = self.attention_projection(enc_hidden)

    # 2.99) Reshape enc_hidden from (batch_size, source_sentence_length, hidden_size * 2)
    # to enc_hidden (batch_size, hidden_size * 2, source_sentence_length)
    enc_hidden = torch.permute(enc_hidden, dims=(0, 2, 1))

    # 3) Run loop through max_target_sentence_length - 1
    timesteps = Y.shape[0]  
    for t in range(timesteps):
      # 3.1) Get the y_t: Tensor of size (batch_size, embedding_size)
      # by taking a slice of Y
      y_t = Y[t]

      # 3.2) Concatenate y_t and o_t to get y_hat_t 
      # y_hat has size (batch_size, embedding_size)
      # o_t has size (batch_size, hidden_size)
      # y_hat_t has size (batch_size, embedding_size + hidden_size)
      y_hat_t = torch.cat((y_t, o_prev), dim=1)

      # 3.21) Reshape y_hat_t to shape (1, batch_size, embedding_size + hidden_size)
      y_hat_t = y_hat_t.unsqueeze(0)

      # 3.3) Pass y_hat_t through decoder LSTM 
      # y_hat_t has size (1, batch_size, embedding_size + hidden_size)
      # dec_state containings dec_hidden (1, batch_size, hidden_size)
      # and dec_cell (1, batch_size, hidden_size)
      # to get dec_hidden_states (1, batch_size, hidden_size)
      # dec_state containing dec_hidden has size (1, batch_size, hidden_size)
      # and dec_cell has size (1, batch_size, hidden_size)
      dec_hidden_states, dec_state = self.decoder_lstm(
          y_hat_t,
          dec_state
      )

      # 3.4) Extract dec_hidden from dec_state
      # dec_hidden size size (1, batch_size, hidden_size)
      dec_hidden, _ = dec_state

      # 3.5) Reshape dec_hidden to (batch_size, hidden_size, 1)
      dec_hidden = dec_hidden.squeeze(0)
      dec_hidden_expanded = dec_hidden.unsqueeze(2)

      # 3.6) Calculate the score by batch multiplying 
      # enc_hidden_proj (batch_size, source_sentence_length, hidden_size)
      # with dec_hidden (batch_size, hidden_size, 1)
      # to get score e_t (batch_size, source_sentence_length, 1)
      e_t = torch.bmm(enc_hidden_proj, dec_hidden_expanded)

      # 3.7) Reshape e_t to get e_t (batch_size, souce_sentence_length)
      e_t = e_t.squeeze(2)

      # 3.8) Mask the <pad> token position with -inf -> They do not contribute the alignment
      with torch.no_grad():
        e_t = e_t.masked_fill(enc_mask, float("-inf"))

      # 3.9) Calculate softmax (with dimension 1) from score to get
      # alpha_t (batch_size, source_sentence_length)
      alpha_t = torch.nn.functional.softmax(e_t, dim=1)

      # 3.11) Expand alpha_t (batch_size, source_sentence_length) to get
      # alpha_t (batch_size, source_sentence_length, 1)
      alpha_t = alpha_t.unsqueeze(2)

      # 3.12) Batch multiplication betweeen
      # enc_hidden (batch_size, hidden_size * 2, source_sentence_length)
      # and alpha_t (batch_size, source_sentence_length, 1)
      # to get weighted average of enc_hidden a_t (batch_size, hidden_size * 2, 1)
      a_t = torch.bmm(enc_hidden, alpha_t)

      # 3.13) Squeeze a_t from (batch_size, hidden_size * 2, 1)
      # to a_t (batch_size, hidden_size * 2)
      a_t = a_t.squeeze(2)

      # 3.14) Concate a_t (batch_size, hidden_size * 2)
      # with dec_hidden  (batch_size, hidden_size) in dimension=1
      # to get u_t (batch_size, hidden_size * 3)
      u_t = torch.cat((a_t, dec_hidden), dim=1)

      # 3.15) Pass through linear layer (v_projection) to 
      # u_t (batch_size, hidden_size * 3)
      # to get v_t (batch_size, hidden_size)
      v_t = self.v_projection(u_t)

      # 3.16) Get o_t (batch_size, hidden) by passing v_t through tanh and
      # then dropout layer
      o_t = self.dropout(torch.tanh(v_t))

      # 3.17) Append o_t to combined_outputs
      combined_outputs.append(o_t)

      # 3.18) Set o_prev to have value of o_t
      o_prev = o_t

    # 4) combined_outputs has t max_target_sentence_length - 1 tensors of size 
    # (batch_size, hidden_size) by stacking
    # -> combined_outputs has size (max_target_sentence_length - 1, batch_size, hidden_size)
    combined_outputs = torch.stack(combined_outputs)

    # 5) Pass through linear layer (target_projection) to get P
    # P has size (max_target_sentence_length - 1, batch_size, vocab_size)
    P = self.target_projection(combined_outputs)

    return P