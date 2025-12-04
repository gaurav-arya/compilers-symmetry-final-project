# Symmetry-Exploiting Compiler Optimizations for MLIR Tensor Programs

## Group Members
Gaurav Arya (gauravar@cmu.edu)  
Apurva Gandhi (apurvag@cs.cmu.edu)

## Project Documents
- [Project Proposal](docs/project_proposal.pdf)
- [Project Milestone](docs/project_milestone.pdf)
- [Project Final Report](docs/project_final_report.pdf)

## Milestone Implementation

* Initial analysis pass to detect symmetry generation and propogation for 2D
tensors (matrices): https://github.com/EnzymeAD/Enzyme-JAX/compare/main...gaurav-
arya:Enzyme-JAX:symmetric-result-analysis
* Recognizing symmetry annotations in input IR and
annotating that the result of transposing a symmetric matrix is still symmetric:
https://github.com/EnzymeAD/Enzyme-JAX/pull/1626
* Annotating that result of A * A will be symmetric if A is symmetric: https://github.com/EnzymeAD/Enzyme-JAX/commit/1659f787d960ef5ae97b20d9afe72d5403921b64
* Successful symmetric detection test on StableHLO MLIR using analysis above: https://github.com/EnzymeAD/Enzyme-JAX/blob/173903fe8d8fd9883a01c3d4dacc13779301a3b4/test/lit_tests/structured_tensors/propagate_symmetric.mlir
* Converting column-major reduction on symmetric matrix to row-major:  https://github.com/EnzymeAD/Enzyme-JAX/commit/1a2a98a8b10efa18fd6a4c133c06154be430268d
* Successful optimization on StableHLO MLIR using optimization above: https://github.com/EnzymeAD/Enzyme-JAX/blob/1a2a98a8b10efa18fd6a4c133c06154be430268d/test/lit_tests/structured_tensors/dot_general_symmetric.mlir

## Final Implementation

In our project, we build on the Enzyme-JAX repo: https://github.com/EnzymeAD/Enzyme-JAX  
Please clone this branch: https://github.com/EnzymeAD/Enzyme-JAX/tree/apga/n_dim_symm_opt.  
To see the changes (commits) that are made by us for our course project:   
```bash
git log --oneline f603104e..HEAD
```

To see the full diffs made for our course project, you can run:
```bash
cd /Users/apurvag/source/Enzyme-JAX && git diff f603104e..HEAD
```

Our analysis pass is implemented in: `/Users/apurvag/source/Enzyme-JAX/src/enzyme_ad/jax/Analysis/PartialSymmetryAnalysis.*`.
Our optimization passes are implemented in `struct TransposePartialSymmetrySimplify` and `struct ReducePartialSymmetryRotateAxes` in `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp`.

Our benchmark is implemented in: `test/benchmark_symmetry.py`.  
To run this benchmark, follow the build from source instructions in the Enzyme-JAX README.md and then pip install the built wheel. Then: 
```python
cd test
python benchmark_symmetry.py
```