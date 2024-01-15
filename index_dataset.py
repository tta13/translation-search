import argparse
from faiss import IndexFlatIP, write_index
from utils import load_dataset, load_embeddings_model
import logging

def parse_args():
  parser = argparse.ArgumentParser(description='Program to index a raw database in a FAISS vector database.')
  parser.add_argument("-p", "--path", help="Database file path.", required=True, dest='path')
  parser.add_argument("-t", "--target", help="Code of the target language to index.", required=True, dest='target')
  parser.add_argument("-s", "--sample", help="Sample size to take.", dest='sample', type=int)
  parser.add_argument("-d", "--device", "Device (like 'cuda' | 'cpu') that should be used for computation.", default="cpu", dest="device")
  parser.add_argument("-o", "--output", help="Output index path.", dest='output')
  args = parser.parse_args()
  return args

def index(path, target, sample, output, device):
  # Load dataset
  logging.info(f'Loading dataset from {path}')
  df = load_dataset(path)
  df = df[[target]]
  df_db = df.iloc[:sample] if sample is not None else df
  
  # Load model
  logging.info('Loading embeddings model')
  model = load_embeddings_model(device=device)
  
  logging.info('Start creating embeddings')
  embeddings = model.encode(df_db[target], batch_size=32, show_progress_bar=True)
  _, d = embeddings.shape
  
  # Add vectors to FAISS
  logging.info('Done! Adding results to FAISS')
  index_target = IndexFlatIP(d)
  index_target.add(embeddings)
  
  # Save index
  logging.info(f'Done! Saving {index_target.ntotal} vector embeddings')
  output = 'out.index' if output is None else output
  write_index(index_target, output)

def main():
  logging.basicConfig(level=logging.INFO)
  args = parse_args()
  index(args.path, args.target, args.sample, args.output, args.device)

if __name__ == '__main__':
  main()
