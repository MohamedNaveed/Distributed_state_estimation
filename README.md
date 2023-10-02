# Distribute State Estimation

# Unifying Consensus and Covariance Intersection for Efficient Distributed State Estimation Over Unreliable Networks

Implementation of a distributed state estimation filter that uses a hybrid approach: it uses iterative covariance intersection to reach consensus over priors, which might become correlated, while consensus over new information is handled using weights based on a Metropolis–Hastings Markov chain. The performance of the hybrid method is evaluated extensively, including comparisons with competing algorithms, with a hypothetical “full history” yardstick, and centralized performance and check for filter consistency using the NEES test. We conduct an assessment on a realistic atmospheric dispersion problem and also on more carefully crafted settings to help characterize particular aspects of the performance.

# Reference: 
A. Tamjidi, R. Oftadeh, M. N. G. Mohamed, D. Yu, S. Chakravorty and D. Shell, "Unifying Consensus and Covariance Intersection for Efficient Distributed State Estimation Over Unreliable Networks," in IEEE Transactions on Robotics, vol. 37, no. 5, pp. 1525-1538, Oct. 2021, doi: 10.1109/TRO.2021.3064102. Link: https://ieeexplore.ieee.org/abstract/document/9400511

Citation (Bibtex):
@ARTICLE{9400511,
  author={Tamjidi, Amirhossein and Oftadeh, Reza and Mohamed, Mohamed Naveed Gul and Yu, Dan and Chakravorty, Suman and Shell, Dylan},
  journal={IEEE Transactions on Robotics}, 
  title={Unifying Consensus and Covariance Intersection for Efficient Distributed State Estimation Over Unreliable Networks}, 
  year={2021},
  volume={37},
  number={5},
  pages={1525-1538},
  doi={10.1109/TRO.2021.3064102}}
