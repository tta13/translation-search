from datasets import load_from_disk

def load_dataset_index(path):
  dataset = load_from_disk(path)
  return dataset['index'].to_pandas()
