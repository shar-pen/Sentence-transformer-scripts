import logging
import json
import sys
import os
import traceback
import argparse
import time
from datetime import datetime

from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator, SequentialEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


# load json file into (anchor, positive, negtive) triplets
def json2triplet(filepath:str):
    l_anchor = []
    l_positive = []
    l_negative = []
    with open(filepath) as file:
        for line in file:
            tmp_dict = json.loads(line)
            anchor = tmp_dict['query']
            for pos in tmp_dict['pos']:
                for neg in tmp_dict['neg']:
                    l_anchor.append(anchor)
                    l_positive.append(pos)
                    l_negative.append(neg)
                    
    data = {
        "anchor": l_anchor,
        "positive": l_positive,
        "negative": l_negative,
    }
    return data

# load json file into (sentence1, sentence2) pairs with float score
def json2pairswithscore(filepath:str):
    
    l_sentences_1 = []
    l_sentences_2 = []
    l_score = []
    
    with open(filepath) as file:
        for line in file:
            tmp_dict = json.loads(line)
            anchor = tmp_dict['query']
            for pos in tmp_dict['pos']:
                l_sentences_1.append(anchor)
                l_sentences_2.append(pos)
                l_score.append(1)
            for neg in tmp_dict['neg']:
                l_sentences_1.append(anchor)
                l_sentences_2.append(neg)
                l_score.append(-1)
                
    data = {
        "sentence1": l_sentences_1,
        "sentence2": l_sentences_2,
        "score": l_score,
    }
    return data

# disables W&B
os.environ["WANDB_DISABLED"] = "true"

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name_or_path', type=str, required=True, help='the path to your model or the HF name of your model')
parser.add_argument('--train_data_path', type=str, required=True, help='the path to your training data')
parser.add_argument('--valid_data_path', type=str, required=True)
parser.add_argument('-o', '--output_dir', type=str, required=False, help='the path to save files', default=os.path.split(os.path.abspath(__file__))[0])

args = parser.parse_args()
print(args)

# parsing args
model_name_or_path = args.model_name_or_path
model_name = os.path.split(model_name_or_path)[-1]

output_dir = args.output_dir
dir_name = f"training-{model_name}-{datetime.now().strftime("%Y%m%d_%H:%M:%S")}"
output_path = os.path.join(output_dir, dir_name)
if not os.path.exists(output_path):
    os.mkdir(output_path)
    
# training args
train_batch_size = 16
num_epochs = 4

    
# logger
logging.basicConfig(filename=os.path.join(output_path, 'training.log'),
                    format="%(asctime)s || %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(args)


# pre actions
t_s = time.time()


# 1.define our SentenceTransformer model

logger.info(f"model path: {os.path.abspath(model_name_or_path)}")
logger.info(f"output path: {output_path}")

model = SentenceTransformer(model_name_or_path)

logger.info(f'model structure: \n {model}')

# 2. loads dataset

triplets = json2triplet(args.train_data_path)
ds_triplets = Dataset.from_dict(triplets)
ds_train = ds_triplets

logger.info(f'training dataset: \n {ds_train}')
logger.info(f'training dataset example: \n {ds_train[0]}')


# ds = ds.train_test_split(test_size=1000)


# 3. define loss

loss_fn = losses.TripletLoss(model=model)

# 4. define evaluator

pairswithscore = json2pairswithscore(args.valid_data_path)
ds_pairswithscore = Dataset.from_dict(pairswithscore)
ds_eval = ds_pairswithscore

logger.info(f'validation dataset: \n {ds_eval}')
logger.info(f'validation dataset example: \n {ds_eval[0]}')

eval_evaluator_pair = EmbeddingSimilarityEvaluator(
    sentences1=ds_eval["sentence1"],
    sentences2=ds_eval["sentence2"],
    scores=ds_eval["score"],
    main_similarity=SimilarityFunction.COSINE,  # similarity function
)

'''
[Example]This is what the output of 'EmbeddingSimilarityEvaluator' looks like:
EmbeddingSimilarityEvaluator: Evaluating the model on the  dataset:
Cosine-Similarity :       Pearson: 0.7874 Spearman: 0.8004
Manhattan-Distance:       Pearson: 0.7823 Spearman: 0.7827
Euclidean-Distance:       Pearson: 0.7824 Spearman: 0.7827
Dot-Product-Similarity:   Pearson: 0.7192 Spearman: 0.7126

[Explain] Both Pearson and Spearman are indicators of how well the model's calculated similarity scores correlate with the ground truth similarity scores from the dataset, but they capture different types of relationships
'''
    
triplets = json2triplet(args.valid_data_path)
ds_triplets = Dataset.from_dict(triplets)
ds_eval = ds_triplets

logger.info(f'validation dataset: \n {ds_eval}')
logger.info(f'validation dataset example: \n {ds_eval[0]}')

eval_evaluator_triplet = TripletEvaluator(
    anchors=ds_eval["anchor"],
    positives=ds_eval["positive"],
    negatives=ds_eval["negative"],
)

'''
[Example]This is what the output of 'TripletEvaluator' looks like:
TripletEvaluator: Evaluating the model on the  dataset:
Accuracy Cosine Distance:   	92.63
Accuracy Dot Product:       	7.30
Accuracy Manhattan Distance:	92.91
Accuracy Euclidean Distance:	92.63

[Explain] 'Accuracy Cosine Distance' shows the accuracy of distinguishing anchor-positive pairs from anchor-negative pairs using cosine similarity. The others are similar but using other similarity metrics. 
Because we use cosine distance, the cosine result is good while the dot result is not. 
'''


# 'SequentialEvaluator' will wrap multiple evaluator together and pass through trainer. 
seq_evaluator = SequentialEvaluator([eval_evaluator_pair, eval_evaluator_triplet])



# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_path,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,   # eval will take some time. Best to keep it less frequent. 
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    logging_steps=200,
    # run_name="sts",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
# model will periodically call evaluator

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    loss=loss_fn,
    evaluator=seq_evaluator,
)
trainer.train()



# 7. Evaluate the model performance on the STS Benchmark test dataset

test_evaluator = seq_evaluator
test_evaluator(model)


# 8. Save the trained & evaluated model locally

final_output_dir = f"{output_path}/final"
model.save(final_output_dir)

# post actions

t_e = time.time()
runtime = t_e - t_s

hours, rem = divmod(runtime, 3600)
minutes, seconds = divmod(rem, 60)

logger.info(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")