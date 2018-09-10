import pickle as pkl


def save_obj(obj, path):
    with open(path, "wb") as writer:
        pkl.dump(obj, writer)


def load_obj(path):
    with open(path, "rb") as reader:
        return pkl.load(reader)
