# TRENCHANT
Experiments for TRENCHANT and baselines' trend predictions

## how to cite
```
@article{doCarmo_ReisFilho_Marcacini_2023, 
	title={TRENCHANT: TRENd PrediCtion on Heterogeneous informAtion NeTworks}, 
	volume={13}, 
	url={https://sol.sbc.org.br/journals/index.php/jidm/article/view/2546}, 
	DOI={10.5753/jidm.2022.2546}, 
	number={6}, 
	journal={Journal of Information and Data Management}, 
	author={do Carmo, P. and Reis Filho, I. J. and Marcacini, R.}, 
	year={2023}, 
	month={Jan.} 
}
```

## GraphEmbeddings
GraphEmbeddings submodule based on https://github.com/shenweichen/GraphEmbedding but the used algorithms works with tf 2.x
### install
inside GraphEmbeddings directory from this repository run
```
python setup.py install
```

## 5W1H
5W1H extraction from https://github.com/fhamborg/Giveme5W1H

## gcn
GCN submodule based on https://github.com/dbusbridge/gcn_tutorial

## metapath2vec
metapath2vec stesubmodule based on https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/metapath2vec-link-prediction.html

## networks
networks with news from http://soybeansandcorn.com/ available at: https://drive.google.com/drive/folders/1WCyJDCWAt2ud_Mu38mZxNUU66B2bEz3M?usp=sharing

## fine-tuned models
fine-tuned models with entire network data available at hugging face with use guides as follows:
https://huggingface.co/paulorvdc/sentencebert-fine-tuned-months-corn
https://huggingface.co/paulorvdc/sentencebert-fine-tuned-months-soy
https://huggingface.co/paulorvdc/sentencebert-fine-tuned-weeks-corn
https://huggingface.co/paulorvdc/sentencebert-fine-tuned-weeks-soy

## wip
updates with improvements new experiments, methods, data and usability features will be posted with the research progress.
we are also looking into publishing fine-tuned models at https://huggingface.co/
