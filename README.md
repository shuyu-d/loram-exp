# LoRAM (Low-rank Additive Model) for the computation of projection from graphs to DAGs

Implementation of LoRAM. Reference: Dong, S. & Sebag, M. (2022). From graphs to DAGs: a low-complexity model and a scalable algorithm. URL https://arxiv.org/abs/2204.04644. 

## Core functions  

- `mf_projdag.py` - Implementation of Algorithm 2 (LoRAM-AGD) for the optimization of LoRAM 
- `splr_expmv.py` - Implementation of Algorithm 1 for (A,C,B) -> (exp(A) odot C) B
- `spmaskmult.pyx` - LoRAM matrix via sparsified low-rank matrix product (Algorithm 3) 


## Requirements

- Python 3+
- `numpy`
- `scipy`
- `NOTEARS/utils.py` - graph simulation, data simulation, and accuracy evaluation from [Zheng et al. 2018](https://github.com/xunzheng/notears)
- `python-igraph`: Install [igraph C core](https://igraph.org/c/) and `pkg-config` first.


## Running a demo
Access the code  loram_exp/

```bash
$ make  # for spmaskmult.pyx 
$ python demo_loram_proj.py
```

