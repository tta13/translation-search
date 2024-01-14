import argparse
import logging
import pandas as pd
from nltk.translate import chrf_score
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

def parse_args():
  parser = argparse.ArgumentParser(description='Program to compute translation metrics on translation entries.')
  parser.add_argument("-i", "--input-path", help="Translation database file path.", required=True, dest='input_path')
  parser.add_argument("-t", "--target", help="Column representing the target on the dataset.", required=True, dest='target')
  parser.add_argument("-r", "--result", help="Column representing the translation result on the dataset.", required=True, dest='result')
  parser.add_argument("-o", "--output", help="Output path of the result.", dest='output_path')
  args = parser.parse_args()
  return args

def chrf_scores(references: list[str], hypothesis: list[str]):
  """Return sentence level chrF scores given a list of references and hyphotesis"""
  return [
    chrf_score.sentence_chrf(ref, hyp)
    for ref, hyp in zip(references, hypothesis)
  ]

def calculate_metrics(input_path, tgt_col, result_col, output_path):
  logging.info(f'Reading dataset {input_path}')
  df_in = pd.read_csv(input_path, sep='\t')

  references=df_in[tgt_col].tolist()
  candidates=df_in[result_col].tolist()

  logging.info(f'Computing chrf scores')
  df_in['chrf_score'] = chrf_scores(references, candidates)

  # Recommended checkpoint  
  logging.info(f'Computing BLEURT')
  checkpoint = 'lucadiliello/BLEURT-20'
  config = BleurtConfig.from_pretrained(checkpoint)
  model = BleurtForSequenceClassification.from_pretrained(checkpoint)
  tokenizer = BleurtTokenizer.from_pretrained(checkpoint)
  model.eval()
  with torch.no_grad():
    inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
    df_in['bleurt'] = model(**inputs).logits.flatten().tolist()
  
  output_path = f'{input_path.split(".")[0]}_scored.csv' if output_path is None else output_path
  logging.info(f'Saving results in {output_path}')
  df_in.to_csv(output_path, sep='\t', index=False)

def main():
  logging.basicConfig(level=logging.INFO)
  args = parse_args()
  calculate_metrics(args.input_path, args.target, args.result, args.output_path)

if __name__ == '__main__':
  main()
