# MaterialsScienceTools.jl

For now this is a random collection of tools for computational materials
science, such as computing elastic moduli or generating elastic fields for
dislocations.

## Summary of Library Content

* whole-space Green's functions for elliptic systems with constant coeffs
   - `CLE.GreenFunction`
   - `CLE.IsoGreenFcn3D`
   - TODO: several generalisations missing

* Dislocations CLE fields (still to be cleaned up!)
   - edge isotropic: `CLE.u_edge_isotropic`
   - edge cubic: `CLE.u_edge_fcc_110`

* Atomistic Dislocation predictors:
   - Pure Edge Dislocation in FCC: `examples/Edge Dislocation FCC 110.ipynb`
   - Pure Edge Dislocation in Si (face-centered diamond-cubic):  `examples/Edge Dislocation Si 110.ipynb`


## TODO

* atomistic Green's functions
* cracks
* higher-order corrections


<!-- [![Build Status](https://travis-ci.org/cortner/MaterialsScienceTools.jl.svg?branch=master)](https://travis-ci.org/cortner/MaterialsScienceTools.jl)

[![Coverage Status](https://coveralls.io/repos/cortner/MaterialsScienceTools.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/cortner/MaterialsScienceTools.jl?branch=master)

[![codecov.io](http://codecov.io/github/cortner/MaterialsScienceTools.jl/coverage.svg?branch=master)](http://codecov.io/github/cortner/MaterialsScienceTools.jl?branch=master)
-->
