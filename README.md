  # Trek Scripts

Use character- and word-level models to generate artificial Star Trek scripts.

## Usage

Install:
```
python setup.py install --user
```

Get help:
```
trek_scripts --help
```

Download transcripts:
```
trek_scripts download --url http://www.chakoteya.net --directory DATA_DIR
```

Strip HTML and reformat transcripts:
```
trek_scripts strip --directory DATA_DIR
```

Encode transcripts for character level models:
```
trek_scripts encode_char --directory DATA_DIR
```

Train character level models:
```
trek_scripts train_char --directory DATA_DIR --model_directory MOD_DIR --hidden_size 512 --num_layers 3 --layer_size 256 --chunk_size 32 --batch_size 64 --epochs 20 --test_size 0.1 --cuda
```

The above are options I've found to work pretty well. Loss will be printed out
as the model trains, and occasionally it will hallucinate an excerpt from a
transcript. If you don't have CUDA on your system, you can drop the `--cuda`
option, but training will be unbearably slow.

Within a few epochs the model should begin printing reasonably structured
transcripts.

If you want to do word level models intead, you'll need to install the
`fasttext` command line program for word level embeddings. See
[here](https://fasttext.cc).

Once you've done that, do
```
trek_scripts fasttext_prep --directory DATA_DIR
```

Then construct word embeddings:
```
trek_scripts fasttext_embed --directory DATA_DIR --word_directory WORD_DIR --epochs 20 --dim 100
```

Generate a word hierarchy:
```
trek_scripts word_nodes --embedding_directory WORD_DIR
```

Encode the transcripts according to those word embeddings:
```
trek_scripts word_embed --directory DATA_DIR --embed_directory WORD_DIR
```

Finally, train a word-level model:
```
trek_scripts train_word --embedding_directory WORD_DIR --num_layers 3 --hidden_size 512 --test_size 0.1 --batch_size 64 --chunk_size 32 --epochs 20 --cuda
```

## References

Trek Scripts uses algorithms from a number of papers, notably:

1. Morin, F., & Bengio, Y. (2005). Hierarchical Probabilistic Neural Network Language Model. Aistats, 5.

2. Mnih, A., & Hinton, G. E. (2008). A Scalable Hierarchical Distributed Language Model. Advances in Neural Information Processing Systems, 1–8. 

3. Enriching Word Vectors with Subword Information, Piotr Bojanowski, Edouard Grave, Armand Joulin and Tomas Mikolov, 2016

