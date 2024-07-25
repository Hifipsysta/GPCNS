# GPCNS

Pytorch implementation for "**Introducing Common Null Space of Gradients for Gradient Projection Methods in Continual Learning**"

## Abstract

Continual learning aims to learn new knowledge from a sequence of tasks without forgetting. Recent studies have found that project-ing gradients onto the orthogonal direction of task-specific features is effective. However, these methods mainly focus on mitigating catastrophic forgetting by adopting old features to construct projection spaces, neglecting the potential to enhance plasticity and the valuable information contained in previous gradients. To enhance plasticity and effectively utilize the gradients from old tasks, we propose Gradient Projection in Common Null Space (GPCNS), which projects current gradients into the common null space of final gradients under all preceding tasks. Moreover, to integrate both feature and gradient information, we propose a collaborative framework that allows GPCNS to be utilized in conjunction with existing gradient projection methods as a plugin that provides gradient information and better plasticity. Experimental evaluations conducted on three benchmarks demonstrate that GPCNS exhibits superior plasticity compared to conventional gradient projection methods. More importantly, GPCNS can effectively improve the backward transfer and average accuracy for existing gradient projection methods when applied as a plugin, which outperforms all the gradient projection methods without increasing learnable parameters and customized objective functions.

## Experiments

#### Datasets

The dataset for CIFAR-100 will be automatically downloaded. For the experiments on MiniImageNet, please download the [train_data](https://drive.google.com/file/d/1fm6TcKIwELbuoEOOdvxq72TtUlZlvGIm/view?usp=sharing) and [test_data](https://drive.google.com/file/d/1RA-MluRWM4fqxG9HQbQBBVVjDddYPCri/view?usp=sharing).

#### Requirements

~~~python
pip install -r requirements.txt
~~~

#### Implementation

~~~python
python main_cf100_GPCNS.py
python main_cf100_FEGPCNS_GPM.py
python main_cf100_FEGPCNS_TRGP.py

python main_mini_GPCNS.py
python main_mini_FEGPCNS_GPM.py
python main_mini_FEGPCNS_TRGP.py

python main_sup_GPCNS.py
python main_sup_FEGPCNS_GPM.py
python main_mini_FEGPCNS_TRGP.py
~~~

