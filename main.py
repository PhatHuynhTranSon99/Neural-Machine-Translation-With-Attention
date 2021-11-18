from language.corpus import DualingualCorpus, map_sentences_inverse

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
