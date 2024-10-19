"""
This is for paired data with float similairity score.
"""

import logging
import sys
import os
import traceback
import argparse
import time
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


# disables W&B
os.environ["WANDB_DISABLED"] = "true"

# parser
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name_or_path', type=str, required=True, help='the path to your model or the HF name of your model')
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
# Semantic Textual Similarity (STS) data
""" Intro
For each sentence pair, we pass sentence A and sentence B through our network which yields the embeddings u und v. The similarity of these embeddings is computed using cosine similarity and the result is compared to the gold similarity score.
e.g. (sentence_A, sentence_B) pairs, with float similarity score
"""
ds = load_dataset("sentence-transformers/stsb")
trainset = ds['train']
evalset = ds['validation']
testset = ds['test']
    

logger.info(f'dataset: \n {ds}')
logger.info(f'dataset example: \n {trainset[0]}')


# 3. define loss
""" Options
In this case, CoSENTLoss is better. 
loss_fn = losses.CosineSimilarityLoss(model=model)
loss_fn = losses.CoSENTLoss(model=model)
"""
loss_fn = losses.CosineSimilarityLoss(model=model)

# 4. define evaluator
eval_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=evalset["sentence1"],
    sentences2=evalset["sentence2"],
    scores=evalset["score"],
    main_similarity=SimilarityFunction.COSINE,  # similarity function
)
    

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
    evaluation_strategy="steps",
    eval_steps=200,
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
    train_dataset=trainset,
    eval_dataset=evalset,
    loss=loss_fn,
    evaluator=eval_evaluator,
)
trainer.train()



# 7. Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=testset["sentence1"],
    sentences2=testset["sentence2"],
    scores=testset["score"],
    main_similarity=SimilarityFunction.COSINE,
)

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