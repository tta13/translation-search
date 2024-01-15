import argparse
import logging
from utils import load_faiss_index, load_dataset, load_embeddings_model
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

def parse_args():
  parser = argparse.ArgumentParser(description='Program to translate text database inputs using translation search.')
  parser.add_argument("--path-db", help="Text database file path.", required=True, dest='path_db')
  parser.add_argument("--path-index", help="FAISS Index file path.", required=True, dest='path_index')
  parser.add_argument("-s", "--source", help="Code of the source language in the database.", required=True, dest='source')
  parser.add_argument("-t", "--target", help="Code of the target language in the database.", required=True, dest='target')
  parser.add_argument("-m", "--model-name", help="Translation model's name.", required=True, dest='model_name')
  parser.add_argument("--sample", help="Sample size to take.", dest='sample', type=int)
  parser.add_argument("-d", "--device", help="Device to run the models. Either 'gpu' or 'cpu'.", default="cpu", dest='device')
  parser.add_argument("-o", "--output", help="Output path of the result.", dest='output_path')
  args = parser.parse_args()
  return args

def translate(path_db, path_index, source, target, device, model_name, sample, output_path):
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
  logging.info(f'Loading models: {model_name} and LaBSE')
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  translation_model = MarianMTModel.from_pretrained(model_name)
  embeddings_model = load_embeddings_model()

  # Inference
  logging.info(f'Translating {df_search.shape[0]} sentences')
  source = df_search[source].tolist()
  target_values = df_search[target].tolist()
  inputs = tokenizer(source, return_tensors="pt", padding=True)  
  if device == 'gpu':
    inputs.to('cuda')
    translation_model.to('cuda')
  translated = translation_model.generate(**inputs)

  # Decoding output
  logging.info('Decoding model output')
  translated_sentences = []
  for t in translated:
    translated_sentences.append(tokenizer.decode(t, skip_special_tokens=True))

  # Compute embeddings for transalated sentences
  logging.info('Compute embeddings for translated sentences')
  queries = embeddings_model.encode(translated_sentences, show_progress_bar=True)

  # Perform the search
  logging.info('Searching index')
  k = 1
  D, I = index_target.search(queries, k)
  results = []
  for row in I:
    results.append(df_index.iloc[row][target].values[0])

  # Save results
  result_df = pd.DataFrame([[w, x, y, z] for w, x, y, z in zip(source, target_values, translated_sentences, results)], columns=['source', 'target', 'translation', 'search_result'])
  output_path = 'result.csv' if output_path is None else output_path
  logging.info(f'Saving results in {output_path}')
  result_df.to_csv(output_path, sep='\t', index=False)

def main():
  logging.basicConfig(level=logging.INFO)
  args = parse_args()
  translate(args.path_db, args.path_index, args.source, args.target, args.device, args.model_name, args.sample, args.output_path)

if __name__ == '__main__':
  main()
