import argparse
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,LabelAccuracyEvaluator,BinaryClassificationEvaluator
from torch.utils.data import DataLoader
import logging
import json
import random
import os
import sys
import math
import numpy as np

random.seed(1)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-pretrain_data", "--pretrain_data", type=str,
                      default="./datasets/groovy/fine_tune.json", help="pre-train data directory")

    args.add_argument("-base_model", "--base_model", type=str,
                      default="LLM/bert-base-uncased", help="base_model")

    args.add_argument("-epoch", "--epoch", type=int,
                      default=100, help="Number of epochs")

    args.add_argument("-batch_size", "--batch_size", type=int,
                      default=32, help="Batch Size")

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default="./output/groovy", help="Folder name to save the models.")

    args = args.parse_args()
    return args

def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content

def IsEnglish(character):
    for cha in character:
        if not 'A' <= cha <= 'Z':
            return False
    else:
        return True

def train(args):
    datapath = args.pretrain_data
    model_save_path = args.outfolder
    train_batch_size = args.batch_size
    num_epochs = args.epoch

    # load data
    data = read_json(datapath)

    # load model
    model_name = args.base_model
    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # encode description
    desc = np.array(data)[:,1].tolist()
    model_nlp = SentenceTransformer(modules=[models.Transformer(model_name), pooling_model])
    desc_embedding = model_nlp.encode(desc, device='cuda', convert_to_numpy=False, convert_to_tensor=True,
                                      batch_size=128)
    con_samples = []
    for i in range(len(data)):
        con_samples.append(InputExample(texts=[data[i][0], data[i][0]], embedding=desc_embedding[i]))

    random.shuffle(con_samples)

    # build dataset
    train_con_dataloader = DataLoader(con_samples, shuffle=True, batch_size=train_batch_size)

    # loss
    train_con_loss = losses.LogNLMultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_con_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # train
    model.fit(train_objectives=[(train_con_dataloader, train_con_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            save_f=True
            )

if __name__ == '__main__':
    args = parse_args()
    train(args)