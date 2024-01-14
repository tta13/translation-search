# translation-search

TCC on translation using search engines techniques.

For each dataset we are using the following translation scheme:
```
a) DGT_TM: English to Spanish (en-es) -> Helsinki-NLP/opus-mt-en-es
b) KDE: German to English (de-en) -> Helsinki-NLP/opus-mt-de-en
c) Global Voices: Portuguese to English (pt-en) ->
d) United Nation: English to French (en-fr) -> Helsinki-NLP/opus-mt-en-fr
```

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
