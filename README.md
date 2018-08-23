# JuLIPMaterials.jl

For now this is a random collection of tools for computational materials
science, such as computing elastic moduli or generating elastic fields for
dislocations.

## Summary of Library Content

* whole-space Green's functions for elliptic systems with constant coeffs
   - 3D Green's function via BBS formula: `CLE.GreenFunction`
   - 3D Isotropic Green's function via explicit formula:`CLE.IsoGreenFcn3D`
   - 3D 1st-order corrector via BBS-like formula: `CLE.GreenFunctionCorrector`

* Dislocations CLE fields (still to be cleaned up!)
   - edge isotropic: `CLE.u_edge_isotropic` or `CLE.IsoEdgeDislocation`
   - edge cubic: `CLE.u_edge_fcc_110`
   - screw isotropic: `CLE.IsoScrewDislocation`
   - general dislocation via BBS method: `CLE.Dislocation`

* Atomistic Dislocation predictors:
   - Pure Edge Dislocation in FCC: `examples/Edge Dislocation FCC 110.ipynb`
   - Pure Edge Dislocation in Si (face-centered diamond-cubic):  `examples/Edge Dislocation Si 110.ipynb`


## TODO

* atomistic Green's functions
* cracks
* higher-order corrections


<!-- [![Build Status](https://travis-ci.org/cortner/JuLIPMaterials.jl.svg?branch=master)](https://travis-ci.org/cortner/JuLIPMaterials.jl)

[![Coverage Status](https://coveralls.io/repos/cortner/JuLIPMaterials.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/cortner/JuLIPMaterials.jl?branch=master)

[![codecov.io](http://codecov.io/github/cortner/JuLIPMaterials.jl/coverage.svg?branch=master)](http://codecov.io/github/cortner/JuLIPMaterials.jl?branch=master)
-->
