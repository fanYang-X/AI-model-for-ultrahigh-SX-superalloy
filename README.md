# AI-model-for-ultrahigh-SX-superalloy
AI accelerates the design of new SX superalloys  

By [Fan Yang](https://github.com/fanYang-X), [Wenyue Zhao](https://shi.buaa.edu.cn/09652/zh_CN/index.htm).

## Background  
It is the ideal goal in the alloy design field to design a new single crystal (SX) superalloy that can serve at ultrahigh temperatures. Here, we propose a new AI-driven alloy development flow. The proposed SaTNC_FT model achieves impressive generalization performance on limited data sets and successfully assists the design of SX alloys for employing at ultrahigh temperatures.

## Updata

***20/1/2024***
Initial commits:

1. Creep dataset, including pre-trained creep datasets (.csv).  
2. Transfer learning
   "pre-trained" is the Pre-trained SaTNC model weights 
   SaTNC model and "transform" for the input of SaTNC are provided

## Usage 

The versions of the pyhton library used are as follows:  
pandas -- 1.3.1  
numpy -- 1.20.3  
scikit-learn -- 1.1.2  
lightgbm -- 3.2.1  
torch -- 1.9.0
