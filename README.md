## Current algorithms:
### Gaussian Elimination
* For N-by-N invertible, dense matrices
* Current speed: `73.52 s` for `N = 2000` on an NVIDIA GeForce 940MX (I think this is still very slow...)
* v2 (May 30, 2023): Was able to improve this to `6.04 s` for `N = 2000` simply by further parallelizing the daxpy (double-precision aX + Y) step to attach a thread to every entry of the matrix as opposed to just every column. I'm not quite sure why this makes such a drastic difference; maybe kernel launches are very expensive.
* Further observations: Measuring how many columns are processed in a second, it looks like the rate falls drastically for larger matrices. I suspect this has to do with the fact that the GEForce 940MX only has ~6000 physical threads, so there must be some sort of context switching happening for large grids. In addition, I found that changing the number of threads per block also has a noticeable impact on performance (I am not totally clear on the reason for this). 
* For `N = 7000`, I had a runtime of about `220 s`, which I still think is very good.
* I also noticed a fair amount of error propogation: For `N = 2000`, the max residual of any component of the solution vector sits under 1e-6, but for `N = 7000` it reaches up to 1e-4. I am not sure how to address this.
* v3 (June 9, 2023): As described in the Miscellaneous section, I am attempting to further speed up this method by caching some repeatedly accessed entries in shared memory. This involves changing the blocks to "tile" the matrix in a column-major (or up-down) fashion. However, this introduced a significant slowdown (`N = 2000` took nearly `10 s`). I strongly believe this is due to the fact that changing the blocks to column-major creates a lot of cache misses (there is a layer of L2 cache that sits between off-chip memory and the SMs), since threads within a block are now jumping around the matrix instead of accessing adjacent elements. So, I got the idea to instead operate on the transpose of the matrix - however, the compromise is that there will be poor spatial locality for swapping rows (`swapRowsKernel()` in the code), since in the transpose this becomes swapping columns. However, there is also the added benefit that `collectCoeffsKernel()` and `findSwapRowKernel()`, which previously operated column-wise, will become row-wise. I'm not entirely sure how the performance will turn out. For now, I'm working on a `USE_TRANSPOSE` option as a macro that will allow easily switching between non-transpose and transpose modes.
* (June 11, 2023): The `USE_TRANSPOSE` option is now working. Operating on the transpose of the matrix did see some performance benefits (with a roughly 2 second advantage over non-transposed on `N = 4096`), but it seems like using shared memory has little effect. If shared memory does have a noticeable effect, we should expect the performance to noticeably improve as the block size increases (since less repeated computation happens), but this does not occur. The improvement seen is probably due to better spatial locality when finding the swap row and computing row scalars for daxpy.
* (July 7, 2023): I wrote a serial implementation of GE and it turns out that my code is actually embarassingly slow; the serial implementation beats it by around 2x on a wide range of N. There is clearly significant work to do to improve the current algorithm.
### Matrix Transpose
* Again, for N-by-N dense matrices
* Currently this is a rather simple algorithm, where threads work only on upper triangular entries and swap them with entries reflected across the diagonal
* Extending this to N-by-M matrices shouldn't be too difficult either - this cannot be done in-place since the output matrix has a different shape (M-by-N), so the best we can do is the same algorithm as above, except writing to a separate output
### Miscellaneous
* (June 8, 2023): I read about a technique called "tiling" [here](https://penny-xu.github.io/blog/tiled-matrix-multiplication), where for many problems you can build an upfront cache in shared memory of the data you need per block, preventing expensive off-chip memory accesses during the algorithm. As I understand, this only provides an advantage when there are intermediate computations on the data you need to perform that use memory accesses (besides the initial read and final write). This happens in the Gaussian code when we perform the daxpy step (we need to read and scale an entry from the row with the current pivot). I will need to change how the blocks are shaped though - currently, they "tile" the matrix in row-major order. Since the entry read/scaled is column-dependent, this will need to change to column-major. 
* (June 9, 2023): My previous understanding of the advantages of tiling was not exactly right - considering the matrix multiplication example in the link more closely, we see that the naive method in fact does no per-thread global memory accesses beyond one read per processed element and a final write, yet somehow tiling reduces this baseline number of accesses. I understand why this happens in this specific case, but still need to think about how the strategy generalizes to other cases. However, the daxpy optimization discussed above should still work (in addition, we can also optimize the computation of row scalars needed for the daxpy step; `collectCoeffsKernel()` in the code).
### More to come soon...
