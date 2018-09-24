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
        "--model", required=True, type=str, choices={"lda", "simple-deep",
                                                     "doc2vec", "ner-model", "ner-comb-model",
                                                     "hi_att", "embedding-glove"}, help="type of model to train")
    parser_train.add_argument("--composition-method", dest="composition_method", required=False,
                              type=str, choices={"avg", "min", "max", "sum"},
                              help="Applicable only for model 'embedding-glove': How to compose embeddings")
    return parser
