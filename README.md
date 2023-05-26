# Demo code of TOAK

This is the demo code for the paper "Adversarial for Social Privacy: A Poisoning Strategy to Degrade User Identity Linkage". It contains the following parts:

* toak_attack.py : source code of proposed TOAK.
* vgae.py : source code of Variational Graph Auto-encoder (VGAE), which is used to generate node embedding.
* dataset/ : a folder for storing dataset. Currently, we provide the Douban dataset and will upload the TF and ARXIV datasets later. The groundtruth/train/test files are used for test the performance of UIL algorithms and not used in TOAK. 

The implemention of UIL algorithms are refered to as their original papers and the code can be found in those papers. We will upload the UIL method code after we check the copyright for sharing them.

The commond for run the TOAK is:

```
python toak_attack.py --dataset=douban  
```


after running, the flipped edge set will be stored at ./attack_graph/douban/toak/
