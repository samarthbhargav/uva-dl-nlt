import logging as log
import json
from os import path

from torch.utils.data import DataLoader

from args_utils import get_argparser
from data_utils.dataloader import ReutersDataset

if __name__ == '__main__':
    args = get_argparser().parse_args()

    log.basicConfig(level=log.DEBUG)

    remove_stopwords = True
    min_freq = 5
    lowercase = True
    
    if args.module == "train":
        dataset = ReutersDataset(args.data_root, "training", remove_stopwords, min_freq, lowercase)
        train_loader = DataLoader(dataset, shuffle=True)
        
        c = 0
        for i in train_loader:
            #print(i)
            
            c =+ 1
            if c%100 == 0:
                print(c)
                    
    else:
        raise ValueError("Unknown module")