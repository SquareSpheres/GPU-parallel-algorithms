# GPU Parallel Algorithms
CUDA algorithms


## Connected components

### ShiloachVishkin Variant
<p>A CUDA implementation of algorithm 1 and 2, published by Guojing Cong and Paul Muzio<br/>
  <a href="https://doi.org/10.1007/978-3-319-14325-5_14">Cong G., Muzio P. (2014) Fast Parallel Connected Components Algorithms on GPUs. In: Lopes L. et al. (eds) Euro-Par 2014: Parallel Processing Workshops. Euro-Par 2014. Lecture Notes in Computer Science, vol 8805. Springer, Cham</a></p>
  
### Stages
<p>A CUDA implementation of algorithm 3 published by Guojing Cong and Paul Muzio<br/>
  <a href="https://doi.org/10.1007/978-3-319-14325-5_14">Cong G., Muzio P. (2014) Fast Parallel Connected Components Algorithms on GPUs. In: Lopes L. et al. (eds) Euro-Par 2014: Parallel Processing Workshops. Euro-Par 2014. Lecture Notes in Computer Science, vol 8805. Springer, Cham</a></p>
  
Note: The header file and source file also contain the two algorithms mentioned above, only with changed return value and function parameters. This is because stages take use of these algorithms in a specific way.

#### TODOS
- <strike>Change from AoS to SoA to increase coalesced memory access</strike> <b>No change</b>
- <strike>Synchronize between kernel calls to check for errors</strike>
- <strike>Set caching strategy, prefer L1 over Shared</strike> <b>No change</b>
- Try to disable/enable L1 cache
- Stages is underperforming. Revise algorithm. 
 
<details> 
  <summary>Results</summary>
  <img src="/results//connected1.PNG">
  <img src="/results//connected2.PNG">
  <img src="/results//connected3.PNG">
</details>
