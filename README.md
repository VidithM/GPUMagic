## Current algorithms:
### Gaussian Elimination
* For N-by-N invertible matrices
* Current speed: `73.52 s` for `N = 2000` on an NVIDIA GeForce 940MX (I think this is still very slow...)
* v2 (May 30, 2023): Was able to improve this to `6.04 s` for `N = 2000` simply by further parallelizing the daxpy (double-precision aX + Y) step to attach a thread to every entry of the matrix as opposed to just every column. I'm not quite sure why this makes such a drastic difference; maybe kernel launches are very expensive.
### More to come soon...
