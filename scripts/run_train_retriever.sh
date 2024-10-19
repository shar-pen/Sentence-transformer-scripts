export CUDA_VISIBLE_DEVICES=2,6

python ./scripts/train_retriever_triplet.py \
    -m '../../cache/officials/bge-large-zh-v1.5' \
    --train_data_path '/data02/hyzhang10/pengxia2/PrivateCollection/Embedding/data/valid_embedding_and_reranker_data.json' \
    --valid_data_path '/data02/hyzhang10/pengxia2/PrivateCollection/Embedding/data/valid_embedding_and_reranker_data.json' 


# python ./scripts/train_retriever_triplet.py \
#     -m '../../cache/officials/bge-large-zh-v1.5' \
#     --train_data_path '/data02/hyzhang10/pengxia2/PrivateCollection/Embedding/data/train_embedding_and_reranker_data.json' \
#     --valid_data_path '/data02/hyzhang10/pengxia2/PrivateCollection/Embedding/data/valid_embedding_and_reranker_data.json' 

# python ./scripts/train_retriever_pair.py \
#     -m '../../cache/officials/bge-m3'