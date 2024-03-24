import argparse
import logging
from utils import load_faiss_index, load_dataset, load_embeddings_model, chunks
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import gc
import torch
from tqdm import tqdm

def parse_args():
  parser = argparse.ArgumentParser(description='Program to translate text database inputs using translation search.')
  parser.add_argument("--path-db", help="Text database file path.", required=True, dest='path_db')
  parser.add_argument("--path-index", help="FAISS Index file path.", required=True, dest='path_index')
  parser.add_argument("-s", "--source", help="Code of the source language in the database.", required=True, dest='source')
  parser.add_argument("-t", "--target", help="Code of the target language in the database.", required=True, dest='target')
  parser.add_argument("-m", "--model-name", help="Translation model's name.", required=True, dest='model_name')
  parser.add_argument("--sample", help="Sample size to take.", dest='sample', type=int)
  parser.add_argument("-d", "--device", help="Device (like 'cuda' | 'cpu') that should be used for computation.", default="cpu", dest='device')
  parser.add_argument("-o", "--output", help="Output path of the result.", dest='output_path')
  args = parser.parse_args()
  return args

def search(path_db, path_index, source, target, device, model_name, sample, output_path):
  # Load index
  logging.info(f'Loading FAISS index: {path_index}')
  index_target = load_faiss_index(path_index)

  # Load database
  logging.info(f'Loading text database: {path_db}')
  dataset = load_dataset(path_db, db='all')
  df = dataset['query'].to_pandas()[[source, target]]
  df_index = dataset['index'].to_pandas()[[target]]
  df_search = df.iloc[:sample, :] if sample is not None else df

  # Load Models
  logging.info(f'Loading model: {model_name}')
  embeddings_model = load_embeddings_model(device=device, model_name=model_name)

	# Inference
  logging.info(f'Searching {df_search.shape[0]} sentences')
  source_sentences = ['query: ' + x for x in df_search[source].tolist()] if model_name=='intfloat/multilingual-e5-base' else df_search[source].tolist() 
  target_sentences = df_search[target].tolist()
  
  # Compute embeddings for transalated sentences
  logging.info('Compute embeddings for sentences')  
  queries = embeddings_model.encode(source_sentences, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

  # Perform the search
  logging.info('Searching index')
  k = 1
  D, I = index_target.search(queries, k)
  results = []
  for row in I:
    results.append(df_index.iloc[row][target].values[0])

  # Save results
  result_df = pd.DataFrame([[w, x, y] for w, x, y in zip(df_search[source].tolist(), target_sentences, results)], columns=['source', 'target', 'search_result'])
  output_path = 'result.csv' if output_path is None else output_path
  logging.info(f'Saving results in {output_path}')
  result_df.to_csv(output_path, sep='\t', index=False)

def main():
  logging.basicConfig(level=logging.INFO)
  args = parse_args()
  search(args.path_db, args.path_index, args.source, args.target, args.device, args.model_name, args.sample, args.output_path)

if __name__ == '__main__':
  main()
