import logging as log

import torch
import numpy as np
import torch.nn.functional as F


def gather_outputs(data, model, cuda, args_model):

    if args_model == 'ner-model':
        threshold = 0.5
        y_true = []
        y_pred = []
        log.info("Gathering outputs")
        with torch.no_grad():

            #count = 0
            for index, (_id, labels, text, ners, _, _) in enumerate(data):

                # count += 1
                # if count == 10:
                #     break

                model.hidden = model.init_hidden()

                if cuda:
                    seq = seq.cuda()

                labels = torch.FloatTensor(labels)

                if len(ners[0]) != 0:
                    ner_word_seq = torch.LongTensor(ners[0])
                    ner_label_seq = torch.LongTensor(ners[1])

                    output = torch.sigmoid(model(ner_word_seq, ner_label_seq))

                    print(output.size())
                    output[output >= threshold] = 1
                    output[output < threshold] = 0

                    y_pred.extend(output.cpu().view(-1).numpy())
                    y_true.extend(labels)

                    if (index + 1) % 100 == 0:
                        log.info("Eval loop: {} done".format(index + 1))

    elif args_model == 'ner-comb-model':
        threshold = 0.5
        y_true = []
        y_pred = []
        log.info("Gathering outputs")
        with torch.no_grad():
            for index, (_id, labels, text, ners, _, _) in enumerate(data):

                model.hidden = model.init_hidden()
                model.hidden_ner = model.init_hidden()

                if cuda:
                    seq = seq.cuda()

                labels = torch.FloatTensor(labels)

                if len(ners[0]) != 0:
                    seq = torch.LongTensor(text)

                    ner_word_seq = torch.LongTensor(ners[0])
                    ner_label_seq = torch.LongTensor(ners[1])

                    output = torch.sigmoid(model(seq, ner_word_seq, ner_label_seq))

                    output[output >= threshold] = 1
                    output[output < threshold] = 0

                    y_pred.append(output.cpu().view(-1).numpy())
                    y_true.append(labels)

                    if (index + 1) % 100 == 0:
                        log.info("Eval loop: {} done".format(index + 1))

    y_true = [np.array(yt) for yt in y_true]
    y_pred = [np.array(yp) for yp in y_pred]
    #y_true, y_pred = np.array(y_true), np.array(y_pred)
    return y_true, y_pred
