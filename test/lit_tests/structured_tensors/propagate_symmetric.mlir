// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_symmetric_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @pass1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<2x2xf32>
  %2 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %3 = stablehlo.add %arg1, %2 : tensor<2x2xf32>
  %4 = stablehlo.add %1, %3 : tensor<2x2xf32>
  %5 = stablehlo.transpose %4, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %5 : tensor<2x2xf32>
}

// CHECK: func.func @pass1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.add %arg0, %0 {enzymexla.guaranteed_symmetric = true} : tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %3 = stablehlo.add %arg1, %2 {enzymexla.guaranteed_symmetric = true} : tensor<2x2xf32>
// CHECK-NEXT:   %4 = stablehlo.add %1, %3 {enzymexla.guaranteed_symmetric = true} : tensor<2x2xf32>
// CHECK-NEXT:   return %4 : tensor<2x2xf32>
// CHECK-NEXT: }
