import torch
from language.corpus import DualingualCorpus, map_sentences_inverse
from dataset.textdataset import CustomDualingualDataset, collate_fn
from model.encoder import Encoder
from model.decoder import Decoder
from model.sequencetosequence import SequenceToSequenceModel
from utils.train import train_model

SOURCE_TRAIN = "data/train.vi"
TARGET_TRAIN = "data/train.en"
SOURCE_EVAL = "data/valid.vi"
TARGET_EVAL = "data/valid.en"
EMBEDDING_SIZE = 5
HIDDEN_SIZE = 10
BATCH_SIZE = 64
DROPOUT = 0.5

if __name__ == "__main__":
    # Create dataloader
    train_corpus = DualingualCorpus(
        source_sentences_path=SOURCE_TRAIN,
        target_sentences_path=TARGET_TRAIN
    )

    eval_corpus = DualingualCorpus(
        source_sentences_path=SOURCE_EVAL,
        target_sentences_path=SOURCE_TRAIN
    )

    print("Source sentence's size:", len(eval_corpus.source_sentences))
    print("Target sentence's size:", len(eval_corpus.target_sentences))

    train_source_vocab = train_corpus.create_vocabulary("source")
    train_target_vocab = train_corpus.create_vocabulary("target")

    # print("Source vocabulary's length: ", len(train_source_vocab))
    # print("Target vocabulary's length: ", len(train_target_vocab))

    train_corpus.convert_words_to_indices(
        source_vocab=train_source_vocab,
        target_vocab=train_target_vocab
    )

    eval_corpus.convert_words_to_indices(
        source_vocab=train_source_vocab,
        target_vocab=train_target_vocab
    )

    train_dataset = CustomDualingualDataset(train_corpus)
    eval_dataset = CustomDualingualDataset(eval_corpus)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=64,
        collate_fn=collate_fn
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=64,
        collate_fn=collate_fn
    )

    # Test the encoder and decoder
    encoder = Encoder(
        vocab=train_source_vocab,
        embed_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE
    )

    decoder = Decoder(
        vocab=train_target_vocab,
        embed_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT
    )

    model = SequenceToSequenceModel(
        encoder=encoder,
        decoder=decoder
    )

    # Train the model
    train_model(model, train_dataloader, eval_dataloader, epochs=1, initial_learning_rate=1.0)