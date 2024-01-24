import argparse
import logging
import pandas as pd
from nltk.translate import chrf_score
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from utils import chunks
from tqdm import tqdm

def parse_args():
  parser = argparse.ArgumentParser(description='Program to compute translation metrics on translation entries.')
  parser.add_argument("-i", "--input-path", help="Translation database file path.", required=True, dest='input_path')
  parser.add_argument("-t", "--target", help="Column representing the target on the dataset.", required=True, dest='target')
  parser.add_argument("-r", "--result", help="Column representing the translation result on the dataset.", required=True, dest='result')
  parser.add_argument("-d", "--device", help="Device (like 'cuda' | 'cpu') that should be used for computation.", default="cpu", dest='device')
  parser.add_argument("-o", "--output", help="Output path of the result.", dest='output_path')
  args = parser.parse_args()
  return args

def chrf_scores(references: list[str], hypothesis: list[str]):
  """Return sentence level chrF scores given a list of references and hyphotesis"""
  return [
    chrf_score.sentence_chrf(ref, hyp)
    for ref, hyp in zip(references, hypothesis)
  ]

def calculate_metrics(input_path, tgt_col, result_col, output_path, device):
  output_path = f'{input_path.split(".")[0]}_scored.csv' if output_path is None else output_path

  logging.info(f'Reading dataset {input_path}')
  df_in = pd.read_csv(input_path, sep='\t')

  references=df_in[tgt_col].tolist()
  candidates=df_in[result_col].tolist()

  if 'chrf_score' not in df_in.columns:
    logging.info(f'Computing chrf scores')
    df_in['chrf_score'] = chrf_scores(references, candidates)
    logging.info(f'Saving partial results in: {output_path}')
    df_in.to_csv(output_path, sep='\t', index=False)

  # Recommended checkpoint
  logging.info(f'Computing BLEURT')
  checkpoint = 'lucadiliello/BLEURT-20'
  config = BleurtConfig.from_pretrained(checkpoint)
  model = BleurtForSequenceClassification.from_pretrained(checkpoint, config=config).to(device).half()
  tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
  model.eval()
  bleurt_scores = []
  with torch.no_grad():
    for refs, cands in tqdm(zip(chunks(references, 32), chunks(candidates, 32)), total=len(references)):
      inputs = tokenizer(refs, cands, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)
      res = model(**inputs).logits.flatten().tolist()
      bleurt_scores.extend(res)
  
    df_in.iloc['bleurt'] = bleurt_scores
    logging.info(f'Saving results in {output_path}')
    df_in.to_csv(output_path, sep='\t', index=False)

def main():
  logging.basicConfig(level=logging.INFO)
  args = parse_args()
  calculate_metrics(args.input_path, args.target, args.result, args.output_path, args.device)

if __name__ == '__main__':
  main()
