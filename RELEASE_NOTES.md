# vectX v0.2.8

A maintenance release focused on code quality, removing dead code, and fixing all
compiler and Clippy warnings across the workspace.

## What's Changed

### Code quality
- Removed an unused `deserialize_vector` helper from the REST layer (the
  `_optional` variant covers all call sites).
- Annotated Qdrant API-compatibility request structs with `#[allow(dead_code)]`
  and clear comments so it's obvious which fields are accepted-for-compat-but-
  not-yet-routed (`UpdateCollectionRequest`, `RecoverSnapshotRequest`,
  `CountRequest`, `SetPayloadRequest`, `DeletePayloadRequest`,
  `ClearPayloadRequest`, `DeleteVectorsRequest`, `BatchSearchRequest`,
  `SearchGroupsRequest`, `DiscoverRequest`, `ContextPair`,
  `DiscoverBatchRequest`, `FacetRequest`, `BatchQueryRequest`,
  `QueryGroupsRequest`, `VectorConfig`).
- Cleaned up dead code in the storage manager and snapshot importer.
- Suppressed dead-method warnings on intentionally-retained `HnswIndex`
  helpers (`contains`, `distance`).
- Removed an unused `HashMap` import from the integration test.

### Clippy fixes (workspace-wide)
- `vectx-core`: removed an unnecessary `Distance::clone()` (it's `Copy`) and
  switched two manual ceiling-division sites to `usize::div_ceil`.
- `vectx-storage`: dropped an empty `if` branch and replaced two
  `Option::map(..).flatten()` calls with `Option::and_then`.
- `vectx-api`: replaced `while let Some(_) = iter.next()` with `for _ in iter`,
  collapsed a nested `if let / else if`, and used `Option::map` where
  appropriate.

### Examples
- Updated `examples/data-chatbot/next-env.d.ts` for the Next.js 16 typed-routes
  layout (`./.next/dev/types/routes.d.ts`).

## Compatibility

No public API or wire-format changes. This is a drop-in replacement for v0.2.7.

## Performance

| Protocol | vs Qdrant | vs Redis |
|----------|-----------|----------|
| Insert (gRPC) | 4.5x faster | 10x faster |
| Search (gRPC) | 6.3x faster | 5x faster |
| Search Latency | 6.6x lower | 5x lower |

## Installation

### Pre-built binaries

Download the appropriate archive for your platform from the assets below.

### From crates.io

```bash
cargo install vectx
```

### From source

```bash
git clone https://github.com/antonellof/vectX.git
cd vectX
cargo build --release
```

## Quick Start

```bash
# Start the server
vectx --http-port 6333 --grpc-port 6334

# Create a collection
curl -X PUT http://localhost:6333/collections/test \
  -H "Content-Type: application/json" \
  -d '{"vectors": {"size": 128, "distance": "Cosine"}}'

# Insert vectors
curl -X PUT http://localhost:6333/collections/test/points \
  -H "Content-Type: application/json" \
  -d '{"points": [{"id": 1, "vector": [0.1, 0.2, ...]}]}'

# Search
curl -X POST http://localhost:6333/collections/test/points/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "limit": 10}'
```

## Links

- **Documentation**: https://docs.rs/vectx
- **Crates.io**: https://crates.io/crates/vectx
- **GitHub**: https://github.com/antonellof/vectX
