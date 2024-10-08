
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

   [![arXiv](https://img.shields.io/badge/arXiv-2409.09573-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2409.09573/) [![ProjectPage](https://img.shields.io/badge/Project_Page-MAICBF-blue)](https://maicbf.github.io/)
  <div align="center"></div>
</p>

<p align="center">
    <img src="./trajectory/maze_2.gif" alt="First GIF" width="400" />
    <img src="./trajectory/ours_trajectory_16_agents_empty_itr_06_fps_10_trailing_random.gif" alt="Second GIF" width="400" />
</p>

## Abstract:

To deploy safe and agile robots in cluttered environments, there is a need to develop fully decentralized controllers that guarantee safety, respect actuation limits, prevent deadlocks, and scale to thousands of agents. Current approaches fall short of meeting all these goals: optimization-based methods ensure safety but lack scalability, while learning-based methods scale but do not guarantee safety. We propose a novel algorithm to achieve safe and scalable control for multiple agents under limited actuation. Specifically, our approach includes: (i) learning a decentralized neural Integral Control Barrier function (neural ICBF) for scalable, input-constrained control, (ii) embedding a lightweight decentralized Model Predictive Control-based Integral Control Barrier Function (MPC-ICBF) into the neural network policy to ensure safety while maintaining scalability, and (iii) introducing a novel method to minimize deadlocks based on gradient-based optimization techniques from machine learning to address local minima in deadlocks. Our numerical simulations show that this approach outperforms state-of-the-art multi-agent control algorithms in terms of safety, input constraint satisfaction, and minimizing deadlocks. Additionally, we demonstrate strong generalization across scenarios with varying agent counts, scaling up to 1000 agents.

### Installation
Create a virtual environment with Anaconda:
```bash
conda create -n maicbf python=3.6
```
Activate the virtual environment:
```bash
source activate maicbf
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
python eval.py --num_agents 1024 --model_path models/agile_u_max_0.2/model_ours_weight_1.0_agents_4_v_max_0.2_u_max_0.2_sigma_0.05_default_iter_69999  --env Maze --vis 1
```
This will evaluate the model for 1024 agents and will detect deadlock, resolve deadlock track collision, resolve collision for all agents using decentralized mpc-icbf controller, save cbf data and time to goal for each agents in a csv file with pretrained weights trained with 4 agents. For pretrained weights see [models](https://github.com/abj247/MA-ICBF/tree/master/models). The evaluation will output the safety rate, number of deadlocks, time taken to complete the simulation. After the simulation ends the output gif will be saved. The evaluation results after completing the simulation will be in the format shown below:


``` python
All collisions resolved.
MPC-ICBF was triggered 2 times to resolve collisions
Steps taken by agents saved to 'steps_taken_by_agents.csv'
GIF saved at: trajectory\ma-icbf_trajectory_16_agents.gif
Evaluation Step: 1 | 1, Time: 5.7216, Deadlocked Agents: 2.0000
Total Number of Collisions : 0.0
collision tracking data saved!!!
Accuracy: [0.97641134, 0.9999202, 0.9374962, 0.9967269, 0.9942482, 2.5885205, 14.096791]
Distance Error (MA-ICBF): 0.6942
Mean Safety Rate (MA-ICBF): 1.0000
Deadlocked agents (MA-ICBF): 2.0000 

```


### Running Baselines

The [baselines](https://github.com/abj247/MA-ICBF/tree/master/baselines) contains the baselines used in the research for comparison. This includes the CBF based learning approaches and MARL algortihms used in the work for comparative studies.


### If you find our work useful, please cite us
```
@article{zinage2024decentralized,
  title={Decentralized Safe and Scalable Multi-Agent Control under Limited Actuation},
  author={Zinage, Vrushabh and Jha, Abhishek and Chandra, Rohan and Bakolas, Efstathios},
  journal={arXiv preprint arXiv:2409.09573},
  year={2024}
}
```


### Acknowledgement

This work is inspired and build upon the work from [macbf](https://github.com/MIT-REALM/macbf) which is the implementation of [Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates](https://arxiv.org/abs/2101.05436). The computational resources for this work are taken from the University of Virginia, Department of Computer Science.


### Reference Links

1. Qin, Z., Zhang, K., Chen, Y., Chen, J. and Fan, C., 2021. Learning safe multi-agent control with decentralized neural barrier certificates.[pdf](https://arxiv.org/abs/2101.05436) [project webpage](https://aeroastro.mit.edu/realm/research-blogs/learning-safe-multi-agent-control-with-decentralized-neural-barrier-certificates/)
2. I.-J. Liu, R. A. Yeh, and A. G. Schwing, “Pic: permutation invariant critic for multi-agent deep reinforcement learning,” in Conference on
Robot Learning. PMLR, 2020, pp. 590–602.[pdf](https://arxiv.org/pdf/1911.00025)
3. S. Nayak, K. Choi, W. Ding, S. Dolan, K. Gopalakrishnan, and H. Balakrishnan, “Scalable multi-agent reinforcement learning through intelligent information aggregation,” in International Conference on Machine Learning. PMLR, 2023, pp. 25 817–25 833.[pdf](https://arxiv.org/pdf/2211.02127)
4. S. Zhang, K. Garg, and C. Fan, “Neural graph control barrier functions guided distributed collision-avoidance multi-agent control,” in Conference on robot learning. PMLR, 2023, pp. 2373–2392.[pdf](https://arxiv.org/pdf/2311.13014)



