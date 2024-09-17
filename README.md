
<p align="center">

  <h2 align="center">Decentralized Safe and Scalable Multi-Agent Control Under Limited Actuation</h2>
  <p align="center">
    <a href="https://vrushabh27.github.io/vrushabh_zinage/"><strong>Vrushabh Zinage</strong></a><sup>1</sup>
    ·
    <a href="https://github.com/abj247"><strong>Abhishek Jha</strong></a><sup>2</sup>
    ·
    <a href="https://engineering.virginia.edu/faculty/rohan-chandra"><strong>Rohan Chandra</strong></a><sup>3</sup>
    ·
    <a href="https://sites.utexas.edu/ebakolas/"><strong>Efstathios Bakolas</strong></a><sup>1</sup>
    
</p>

<p align="center">
    <sup>1</sup>University of Texas at Austin · <sup>2</sup> Delhi Technological University · <sup>3</sup>University of Virginia
</p>
   <h3 align="center">

   [![arXiv](https://img.shields.io/badge/arXiv-2408.10154-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2409.09573/) [![ProjectPage](https://img.shields.io/badge/Project_Page-MAICBF-blue)](https://maicbf.github.io/)
  <div align="center"></div>
</p>

## Abstract:

In this paper, we consider the problem of safe and deadlock-free control for multiple agents operating in an environment with obstacles and other agents under limited actuation. We propose a joint-learning framework where the decentralized control policy and Integral Control Barrier Functions (ICBF), which act as safety certificates for input-constrained nonlinear systems, are learned simultaneously. As the neural control policy provides only probabilistic guarantees for collision avoidance, we embed a novel computationally light decentralized MPC Integral Control Barrier Functions (MPC-ICBF) into the neural network policy to guarantee safety as well as scalability. By drawing an analogy between minimizing non-convex loss functions by well-established gradient methods in machine learning and the problem of local minima, we propose a novel approach that is augmented with the proposed learning framework to minimize deadlocks. Numerical simulations demonstrate that our approach outperforms other state of the art multi-agent control algorithms in terms of safety, input constraint satisfaction, and number of deadlocks. Furthermore, we show good generalization capabilities across different scenarios with varying numbers of agents up to 1000 agents.

### Installation
Create a virtual environment with Anaconda:
```bash
conda create -n maicbf python=3.6
```
Activate the virtual environment:
```bash
sourve activate maicbf
```
Clone this repository:
```bash
git clone https://github.com/abj247/maicbf.git
```
Enter the main folder and install the dependencies:
```bash
pip install -r requirements.txt
```
### Training
To train the ma-icbf model specified number of agents (e.g. 4) use this command,
```bash
python train.py --num_agents 4 
```
This will train the model with 4 agents 


### Evaluation
For evaluation of ma-icbf model use this command,
```bash
python eval.py --num_agents 1024 --model_path <path_to_model>  --env Maze --vis 1
```
This will evaluate the model for 1024 agents and will detect deadlock, resolve deadlock track collision, resolve collision for all agents using mpc-cbf controller save cbf data and time to goal for each agents in a csv file for pretrained weights see [models](https://github.com/abj247/MA-ICBF/tree/master/models).





## Acknowledgement

This work is inspired and build upon the work from [macbf](https://github.com/MIT-REALM/macbf) which is the implementation of  [Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates](https://arxiv.org/abs/2101.05436). The compute resource for this work is taken from the University of Virginia, Department of Computer Science cluster.


## Reference Links

1. Qin, Z., Zhang, K., Chen, Y., Chen, J. and Fan, C., 2021. Learning safe multi-agent control with decentralized neural barrier certificates.[pdf](https://arxiv.org/abs/2101.05436) [project webpage](https://aeroastro.mit.edu/realm/research-blogs/learning-safe-multi-agent-control-with-decentralized-neural-barrier-certificates/)
