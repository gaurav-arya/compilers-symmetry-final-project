  // RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=transpose_symmetric_simplify" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @pass1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = stablehlo.reshape %arg0 {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = stablehlo.reshape %arg1 {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %2 = stablehlo.add %0, %1 : tensor<2x2xf32>
  %3 = stablehlo.transpose %2, dims = [1, 0] : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %3 : tensor<2x2xf32>
}

// CHECK: func.func @pass1(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
// CHECK-NEXT:   %0 = stablehlo.reshape %arg0 {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %1 = stablehlo.reshape %arg1 {enzymexla.guaranteed_symmetric = true} : (tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 {enzymexla.guaranteed_symmetric = true} : tensor<2x2xf32>
// CHECK-NEXT:   return %2 : tensor<2x2xf32>
// CHECK-NEXT: }
