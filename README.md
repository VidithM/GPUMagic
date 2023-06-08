## Current algorithms:
### Gaussian Elimination
* For N-by-N invertible, dense matrices
* Current speed: `73.52 s` for `N = 2000` on an NVIDIA GeForce 940MX (I think this is still very slow...)
* v2 (May 30, 2023): Was able to improve this to `6.04 s` for `N = 2000` simply by further parallelizing the daxpy (double-precision aX + Y) step to attach a thread to every entry of the matrix as opposed to just every column. I'm not quite sure why this makes such a drastic difference; maybe kernel launches are very expensive.
* Further observations: Measuring how many columns are processed in a second, it looks like the rate falls drastically for larger matrices. I suspect this has to do with the fact that the GEForce 940MX only has ~6000 physical threads, so there must be some sort of context switching happening for large grids. In addition, I found that changing the number of threads per block also has a noticeable impact on performance (I am not totally clear on the reason for this). 
* For `N = 7000`, I had a runtime of about `220 s`, which I still think is very good.
* I also noticed a fair amount of error propogation: For `N = 2000`, the max residual of any component of the solution vector sits under 1e-6, but for `N = 7000` it reaches up to 1e-4. I am not sure how to address this.
### Matrix Transpose
* Again, for N-by-N dense matrices
* Currently this is a rather simple algorithm, where threads work only on upper triangular entries and swap them with entries reflected across the diagonal
* Extending this to N-by-M matrices shouldn't be too difficult either - this cannot be done in-place since the output matrix has a different shape (M-by-N), so the best we can do is the same algorithm as above, except writing to a separate output
### Miscellaneous
* (June 8, 2023): I read about a technique called "tiling" [here](https://penny-xu.github.io/blog/tiled-matrix-multiplication), where for many problems you can build an upfront cache in shared memory of the data you need per block, preventing expensive off-chip memory accesses during the algorithm. As I understand, this only provides an advantage when there are intermediate computations on the data you need to perform (besides the initial read and final write). This happens in the Gaussian code when we perform the daxpy step (we need to do a division). I will need to change how the blocks are shaped though - currently, they "tile" the matrix in row-major order. Since the value used in the division is column-dependent, this will need to change to column-major. 
### More to come soon...
