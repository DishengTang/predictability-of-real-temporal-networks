# Predictability of real temporal networks [[NSR](https://doi.org/10.1093/nsr/nwaa015),[arXiv](https://arxiv.org/abs/2007.04828)] 




## Introduction

Links in most real networks often change over time. Such temporality of links encodes the ordering and causality of interactions between nodes and has a profound effect on network dynamics and function. Empirical evidence has shown that the temporal nature of links in many real-world networks is not random. Nonetheless, it is challenging to predict temporal link patterns while considering the entanglement between topological and temporal link patterns. 

In this paper, we propose an entropy-rate-based framework, based on combined topological–temporal regularities, for quantifying the predictability of any temporal network. We apply our framework on various model networks, demonstrating that it indeed captures the intrinsic topological–temporal regularities whereas previous methods considered only temporal aspects. We also apply our framework on 18 real networks of different types and determine their predictability. 

Interestingly, we find that, for most real temporal networks, despite the greater complexity of predictability brought by the increase in dimension, the combined topological–temporal predictability is higher than the temporal predictability. Our results demonstrate the necessity for incorporating both temporal and topological aspects of networks in order to improve predictions of dynamical processes.


#### Paper link: [Predictability of Real Temporal Networks](https://doi.org/10.1093/nsr/nwaa015)


## Running the experiments

### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
numpy==1.20.0
networkx==2.5.1
```

### Dataset and Preprocessing

#### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```raw_data/```.

#### Preprocess the data
We use a 2-dimensional matrix M to represent temporal networks. For large networks, we extract the subgraph of active nodes and then remove the inactive links in M to reduce computational cost. All calculations are performed base on the obtained matrix M_tilde. Check the Methods Section of the paper for details.
```



### Predictability calculation

Topological-temporal predictability (TTP):
```{bash}
# The College-Forum dataset
python main.py -dp '../raw_data/forum.txt' -l 'source target time' -fn ./TTP/'CF'.txt -np 100 -n 0 -conv 1 -dr 1 -fre 0 -cc 0 -ft 1
```

Normalized Topological-temporal predictability (NTTP):
```{bash}
# The Manufacturing-Emails dataset
python main.py -dp '../raw_data/manufacturingEmails.txt' -l 'source target weight time' -fn ./TTP/'ME'.txt -np 100 -n 1 -nb 100 -conv 1 -dr 1 -fre 0 -cc 0 -ft 1
```

#### Note: We prepared bash files for College-Forum, College-Message, Manufacturing-Emails and Reality-Mining datasets in the script folder.


## Cite us
If you use this code as part of any research for publication, please acknowledge the following paper:
[Predictability of Real Temporal Networks](https://doi.org/10.1093/nsr/nwaa015)

```bibtex
@article{tang2020predictability,
  title={Predictability of real temporal networks},
  author={Tang, Disheng and Du, Wenbo and Shekhtman, Louis and Wang, Yijie and Havlin, Shlomo and Cao, Xianbin and Yan, Gang},
  journal={National Science Review},
  volume={7},
  number={5},
  pages={929--937},
  year={2020},
  publisher={Oxford University Press}
}
```


