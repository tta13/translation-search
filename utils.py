from datasets import load_from_disk
from faiss import read_index
from sentence_transformers import SentenceTransformer

model_name = 'LaBSE'

def load_dataset_index(path):
  dataset = load_from_disk(path)
  return dataset['index'].to_pandas()

def load_faiss_index(path):
  return read_index(path)

def load_embeddings_model():
  return SentenceTransformer(model_name)
