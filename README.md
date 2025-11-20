# Symmetry-Exploiting Compiler Optimizations for MLIR Tensor Programs

## Group Members
Gaurav Arya (gauravar@cmu.edu)  
Apurva Gandhi (apurvag@cs.cmu.edu)

## Project Documents
- [Project Proposal](docs/project_proposal.pdf)

## Implementation

* Initial analysis pass to detect symmetry generation and propogation for 2D
tensors (matrices): https://github.com/EnzymeAD/Enzyme-JAX/compare/main...gaurav-
arya:Enzyme-JAX:symmetric-result-analysis
* Recognizing symmetry annotations in input IR and
annotating that the result of transposing a symmetric matrix is still symmetric:
https://github.com/EnzymeAD/Enzyme-JAX/pull/1626
* Annotating that result of A & A will be symmetric if A is symmetric: https://github.com/EnzymeAD/Enzyme-JAX/commit/
1659f787d960ef5ae97b20d9afe72d5403921b64
* Converting column-major reduction on symmetric matrix to row-major:  https://github.
com/EnzymeAD/Enzyme-JAX/commit/1a2a98a8b10efa18fd6a4c133c06154be430268d
