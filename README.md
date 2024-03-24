# Search and retrieval in translation memories using machine translation and semantic search

Translation is more than just changing words from one language to another, it builds bridges between cultures, it is a gesture of empathy that allows people to experience cultural phenomena from another culture’s lens. In this scenario, enter Translation Memory systems (TMs), whose core function lies in matching and retrieving existing translations in their databases to assist translators in new translations. However, this crucial process remains restricted by algorithms that depend on edit distance, the number of deletions, insertions or substitutions required to transform the source segment into the target segment, which presents a significant limitation for modern TMs. This work introduces an alternative translation memory system paradigm based machine translation models and semantic sentence embeddings to surpass state-of-the-art TMs by improving the capture of semantic similarity between translated text segments. The results obtained demonstrate that the strategy is effective and promising, with significantly superior performance than tested alternatives (P<0.05) in translations from English to Spanish and French, as well as from German and Portuguese to English, but could be extended to any language pair.

## System requirements
- [Conda](https://anaconda.org/anaconda/conda)
- [Python 3](https://www.python.org/)

## Installing dependencies
On CPU:
```bash 
$ conda env create --prefix ./env --file environment_cpu.yml
```

On GPU:
```bash 
$ conda env create --prefix ./env --file environment_gpu.yml
```

## Running the commands
```bash
# Change to src directory
$ cd src

# Run Indexing of dataset target language sentences
$ python index_dataset.py --path path/to/db.hf --target en --device cuda --model-name intfloat/multilingual-e5-base --output path/to/your_index.index

# Run translation pipeline (automatic machine translation + semantic search)
$ python translation.py --path-db path/to/db.hf --source pt --target en --device cuda --path-index path/to/your_index.index --model-name Helsinki-NLP/opus-mt-ROMANCE-en --output path/to/results/result.csv

# Run translation pipeline (semantic search only)
$ python semantic_search.py --path-db path/to/db.hf --path-index path/to/your_index.index --source pt --target en --model-name intfloat/multilingual-e5-base --device cuda --output path/to/results/result.csv

# Run evaluation metrics on translated results 
$ python metrics.py --input-path path/to/results/result.csv --target target --result search_result --device cuda --output path/to/results/scored.csv
```
