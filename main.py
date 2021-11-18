import torch
from language.corpus import DualingualCorpus, map_sentences_inverse
from dataset.textdataset import CustomDualingualDataset, collate_fn

SOURCE_TRAIN = "data/train.vi"
TARGET_TRAIN = "data/train.en"

if __name__ == "__main__":
    train_corpus = DualingualCorpus(
        source_sentences_path=SOURCE_TRAIN,
        target_sentences_path=TARGET_TRAIN
    )

    train_source_vocab = train_corpus.create_vocabulary("source")
    train_target_vocab = train_corpus.create_vocabulary("target")

    print("Source vocabulary's length: ", len(train_source_vocab))
    print("Target vocabulary's length: ", len(train_target_vocab))

    train_corpus.convert_words_to_indices(
        source_vocab=train_source_vocab,
        target_vocab=train_target_vocab
    )

    train_dataset = CustomDualingualDataset(train_corpus)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=64,
        collate_fn=collate_fn
    )

    source_sentences, target_sentences, source_length = next(iter(train_dataloader))

    print(source_sentences.shape)
    print(target_sentences.shape)
    print(source_length)

    test_source_sentences = [source_sentences[:, 0].tolist()]
    test_target_sentences = [target_sentences[:, 0].tolist()]

    print(map_sentences_inverse(test_source_sentences, train_source_vocab))
    print(map_sentences_inverse(test_target_sentences, train_target_vocab))