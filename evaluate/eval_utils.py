import logging as log

import torch
import numpy as np


def gather_outputs(data, model, cuda):
    threshold = 0.5
    y_true = []
    y_pred = []
    log.info("Gathering outputs")
    with torch.no_grad():
        for index, (_id, labels, text, _,  _, _) in enumerate(data):
            
            model.hidden = model.init_hidden()
            seq = torch.LongTensor(text)
            if cuda:
                seq = seq.cuda()
            output = F.sigmoid(model(seq))
            output[output >= threshold] = 1
            output[output < threshold] = 0

            y_pred.append(output.cpu().view(-1).numpy())
            y_true.append(labels)

            if (index + 1) % 1000 == 0:
                log.info("Eval loop: {} done".format(index + 1))
            
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return y_true, y_pred
