# Grand Pattern — PTX Implementation

**NVIDIA PTX ISA (Parallel Thread Execution Assembly) kernels** for the Fibonacci Dual-Direction Architecture.

This is the lowest-level GPU compute implementation — hand-written PTX assembly with CUDA Driver API host code.

## Architecture

The GPU kernels handle the parallel compute-heavy operations of the Grand Pattern:

- **Embedding operations**: cosine similarity, centroid, distance — massively parallel across embeddings
- **JEPA prediction**: batch prediction across all rooms simultaneously
- **Double-entry balance**: parallel reduction to check all rooms balance
- **GC merge**: parallel similarity computation → merge pairs
- **Vibe computation**: parallel reduction across perception/prediction DBs
- **Cross-room correlation**: matrix of all-pairs cosine similarity

## Kernels

| Kernel | Description |
|--------|-------------|
| `cosine_similarity` | Parallel cosine similarity between two embedding vectors |
| `batch_predict` | Each thread handles one embedding prediction |
| `balance_check` | Parallel reduction comparing perception vs prediction counts |
| `decay` | Element-wise exp(-rate * age) on strengths |
| `vibe_compute` | Parallel vibe computation from embeddings + velocities |
| `correlation_matrix` | All-pairs cosine similarity matrix across rooms |
| `merge_candidates` | Identify merge candidates above similarity threshold |

## Building

```bash
mkdir build && cd build
cmake ..
make
```

Requires:
- NVIDIA CUDA Toolkit 12+
- A CUDA-capable GPU (compute capability 7.0+)

## Running Tests

```bash
./build/test_ptx_kernels
```

## Double-Entry on GPU

Each thread handles one room. `perception_count[i]` must equal `prediction_count[i]`. Atomic reduction counts imbalances. The balance check kernel enforces the double-entry bookkeeping invariant across the entire room graph.

## PTX Details

- Manual register management with `.reg` directives
- Shared memory for warp-level reductions
- `bar.sync` for block-level synchronization
- `atom.add` for global atomic operations
- Direct load/store from global memory with cache hints

## License

MIT
