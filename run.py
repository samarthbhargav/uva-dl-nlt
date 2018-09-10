import logging as log

from torch.utils.data import DataLoader

from args_utils import get_argparser
from data_utils.dataloader import ReutersDataset

if __name__ == '__main__':
    args = get_argparser().parse_args()

    log.basicConfig(level=log.DEBUG)

    if args.module == "train":
        dataset = ReutersDataset(args.data_root, "training")
        train_loader = DataLoader(dataset, shuffle=True)
        for i in train_loader:
            print(i)
    else:
        raise ValueError("Unknown module")