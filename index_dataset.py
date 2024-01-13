import argparse
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatIP, write_index
from utils import load_dataset_index
import logging

# carregar modelo
model_name = 'LaBSE'
model = SentenceTransformer(model_name)

def parse_args():
  parser = argparse.ArgumentParser(description='Program to index a raw database in a FAISS vector database.')
  parser.add_argument("-p", "--path", help="Database file path.", required=True, dest='path')
  parser.add_argument("-t", "--target", help="Code of the target language to index.", required=True, dest='target')
  parser.add_argument("-s", "--sample", help="Sample size to take.", dest='sample', type=int)
  parser.add_argument("-o", "--output", help="Output index path.", dest='output')
  args = parser.parse_args()
  return args

def index(path, target, sample, output):
  logging.info(f'Loading dataset from {path}')
  df = load_dataset_index(path)
  df = df[[target]]
  df_db = df.iloc[:sample] if sample is not None else df
  logging.info('Start creating embeddings')
  embeddings = model.encode(df_db[target], batch_size=32, show_progress_bar=True)
  _, d = embeddings.shape
  logging.info('Done! Adding results to FAISS')
  index_target = IndexFlatIP(d)         # o index é produto interno, mas busca pelo cosseno com os vetores já normalizados
  index_target.add(embeddings)          # adiciona target ao index
  # Save index
  logging.info(f'Done! Saving {index_target.ntotal} vector embeddings')
  output = 'out.index' if output is None else output
  write_index(index_target, output)

def main():
  logging.basicConfig(level=logging.INFO)
  args = parse_args()
  index(args.path, args.target, args.sample, args.output)

if __name__ == '__main__':
  main()
