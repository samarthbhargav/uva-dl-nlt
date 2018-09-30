import argparse


def get_argparser():
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--verbose', dest='verbose', required=False,
                        default=False, action='store_true', help="set to true for verbose output")
    parser.add_argument('--seed', dest="seed", required=False,
                        default=42, help="random seed")

    subparsers = parser.add_subparsers(dest='module', help='module to run')

    # subparaser for building permit-desc classifiers
    parser_train = subparsers.add_parser(
        "train", help="train the model, given parameters")
    parser_train.add_argument(
        "--data-root", dest="data_root", type=str, required=True, help="location of data")
    parser_train.add_argument(
        "--model", required=True, type=str, choices={"han", "lda", "simple-deep",
                                                     "doc2vec", "ner-model", "ner-comb-model",
                                                     "hi_att", "embedding-glove"}, help="type of model to train")
    parser_train.add_argument("--model-id", required=True, type=str,
                              help="ID of the model. Used for persisting results")
    parser_train.add_argument("--epochs", type=int, default=10,
                              help="Number of epochs to run. Applicable only to some models (deep)")
    parser_train.add_argument("--composition-method", dest="composition_method", required=False,
                              type=str, choices={"avg", "min", "max", "sum"},
                              help="[embedding-glove] How to compose embeddings")
    parser_train.add_argument("--n-layers", dest="n_layers", type=int, default=1,
                              help="[simple-deep] Number of layers to use in LSTM")
    parser_train.add_argument("--bi-directional", action="store_true", default=False,
                              dest="bidirectional", help="[simple-deep] Use bi-directional LSTM ")
    parser_train.add_argument("--dropout", type=float, default=0.3,
                              help="[simple-deep] (variational) Dropout (resused in LSTM/Fully connected as well)")

    parser_train.add_argument("--num-topics",
                              dest="num_topics",
                              type=int,
                              required=False,
                              default=100,
                              help="number of topics for lda model")

    return parser
