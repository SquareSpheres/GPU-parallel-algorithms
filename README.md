# GPU Parallel Algorithms
CUDA algorithms


## Connected components

### ShiloachVishkin Variant
<p>A CUDA implementation of an algorithm published by Guojing Cong and Paul Muzio<br/>
  <a href="https://doi.org/10.1007/978-3-319-14325-5_14">Cong G., Muzio P. (2014) Fast Parallel Connected Components Algorithms on GPUs. In: Lopes L. et al. (eds) Euro-Par 2014: Parallel Processing Workshops. Euro-Par 2014. Lecture Notes in Computer Science, vol 8805. Springer, Cham</a></p>

#### TODOS
- <strike>Change from AoS to SoA to increase coalesced memory access</strike> <b>No change</b>
- <strike>Synchronize between kernel calls to check for errors</strike>
- Set caching strategy, prefer L1 over Shared
- Try to disable/enable L1 cache
 
<details> 
  <summary>Results</summary>
  <img src="/results//connected1.PNG">
  <img src="/results//connected2.PNG">
  <img src="/results//connected3.PNG">
</details>
