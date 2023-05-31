## Current algorithms:
### Gaussian Elimination
* For N-by-N invertible matrices
* Current speed: `73.52 s` for `N = 2000` on an NVIDIA GeForce 940MX (I think this is still very slow...)
* v2 (May 30, 2023): Was able to improve this to `6.04 s` for `N = 2000` simply by further parallelizing the daxpy (double-precision aX + Y) step to attach a thread to every entry of the matrix as opposed to just every column. I'm not quite sure why this makes such a drastic difference; maybe kernel launches are very expensive.
* Further observations: Measuring how many columns are processed in a second, it looks like the rate falls drastically for larger matrices. I suspect this has to do with the fact that the GEForce 940MX only has ~6000 physical threads, so there must be some sort of context switching happening for large grids. In addition, I found that changing the number of threads per block also has a noticeable impact on performance (I am not totally clear on the reason for this). 
* For `N = 7000`, I had a runtime of about `220 s`, which I still think is very good.
### More to come soon...
