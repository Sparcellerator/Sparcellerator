// NOTE: This test requires gpu-sm80 and CUDA 11+ for Blocked-ELL support
//
// DEFINE: %{compile} = mlir-opt %s \
// DEFINE:   --sparsifier="enable-gpu-libgen gpu-triple=nvptx64-nvidia-cuda gpu-chip=sm_80 gpu-features=+ptx71 gpu-format=bell
// DEFINE: %{run} = mlir-runner \
// DEFINE:   --shared-libs=%mlir_cuda_runtime \
// DEFINE:   --shared-libs=%mlir_c_runner_utils \
// DEFINE:   --e main --entry-point-result=void \
// DEFINE: | FileCheck %s

#BELL = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : blocked_ell[2,4]), // 2x2 blocks, 4 cols/row
  posWidth = 32,
  crdWidth = 32
}>

module {
  llvm.func @mgpuCreateSparseEnv()
  llvm.func @mgpuDestroySparseEnv()

  // Computes C = A x B with A sparse BELL
  func.func @matmulBELL(%A: tensor<8x8xf32, #BELL>,
                        %B: tensor<8x8xf32>,
                        %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %D = linalg.matmul
      ins(%A, %B: tensor<8x8xf32, #BELL>, tensor<8x8xf32>)
      outs(%C: tensor<8x8xf32>) -> tensor<8x8xf32>
    return %D: tensor<8x8xf32>
  }

  func.func @dump(%mat: tensor<8x8xf32>) {
    %f0 = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %v = vector.transfer_read %mat[%i, %c0], %f0 : tensor<8x8xf32>, vector<8xf32>
      vector.print %v : vector<8xf32>
    }
    return
  }

  func.func @main() {
    llvm.call @mgpuCreateSparseEnv(): () -> ()
    
    // Stress test with a dense matrix DA.
    %DA = tensor.generate {
    ^bb0(%i: index, %j: index):
      %k = arith.addi %i, %j : index
      %l = arith.index_cast %k : index to i64
      %f = arith.uitofp %l : i64 to f32
      tensor.yield %f : f32
    } : tensor<8x8xf32>
    
    // Convert to Blocked-ELL format
    %Abell = sparse_tensor.convert %DA : tensor<8x8xf32> to tensor<8x8xf32, #BELL>
    
    // Initialize output matrix
    %C0 = tensor.generate {
    ^bb0(%i: index, %j: index):
      %zero = arith.constant 0.0 : f32
      tensor.yield %zero : f32
    } : tensor<8x8xf32>

    // Perform Blocked-ELL SpMM
    %Result = call @matmulBELL(%Abell, %DA, %C0)
      : (tensor<8x8xf32, #BELL>, tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    
    // Verify results
    // CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0 )
    // CHECK-NEXT: ( 0, 28, 56, 84, 112, 140, 168, 196 )
    // CHECK-NEXT: ( 0, 56, 112, 168, 224, 280, 336, 392 )
    // CHECK-NEXT: ( 0, 84, 168, 252, 336, 420, 504, 588 )
    // CHECK-NEXT: ( 0, 112, 224, 336, 448, 560, 672, 784 )
    // CHECK-NEXT: ( 0, 140, 280, 420, 560, 700, 840, 980 )
    // CHECK-NEXT: ( 0, 168, 336, 504, 672, 840, 1008, 1176 )
    // CHECK-NEXT: ( 0, 196, 392, 588, 784, 980, 1176, 1372 )
    call @dump(%Result) : (tensor<8x8xf32>) -> ()
    
    // Cleanup
    bufferization.dealloc_tensor %Abell : tensor<8x8xf32, #BELL>
    llvm.call @mgpuDestroySparseEnv(): () -> ()
    return
  }
}

