use actix_web::{web, App, HttpServer, HttpResponse, Result as ActixResult};
use actix_cors::Cors;
use actix_files::Files;
use actix_multipart::Multipart;
use chrono::Utc;
use vectx_core::{CollectionConfig, Collection, Distance, Point, PointId, Vector, PayloadFilter, FilterCondition, Filter, MultiVector};
use vectx_storage::StorageManager;
use serde::{Deserialize, Deserializer, Serialize};
use std::sync::Arc;
use std::path::Path;
use std::collections::HashMap;
use std::time::Instant;
use futures_util::StreamExt;

/// Create Qdrant-compatible JSON response with status and time
fn qdrant_response<T: Serialize>(result: T, start_time: Instant) -> HttpResponse {
    let elapsed = start_time.elapsed().as_secs_f64();
    HttpResponse::Ok().json(serde_json::json!({
        "result": result,
        "status": "ok",
        "time": elapsed
    }))
}

/// Create Qdrant-compatible error response
fn qdrant_error(error: &str, start_time: Instant) -> HttpResponse {
    let elapsed = start_time.elapsed().as_secs_f64();
    HttpResponse::BadRequest().json(serde_json::json!({
        "status": {
            "error": error
        },
        "time": elapsed
    }))
}

/// Create Qdrant-compatible not found response
fn qdrant_not_found(error: &str, start_time: Instant) -> HttpResponse {
    let elapsed = start_time.elapsed().as_secs_f64();
    HttpResponse::NotFound().json(serde_json::json!({
        "status": {
            "error": error
        },
        "time": elapsed
    }))
}

// Dashboard configuration
const STATIC_DIR: &str = "./static";
const DASHBOARD_PATH: &str = "/dashboard";

#[derive(Deserialize)]
struct CreateCollectionRequest {
    /// Dense vectors configuration (optional - can be omitted for sparse-only collections)
    #[serde(default, deserialize_with = "deserialize_vectors_optional")]
    vectors: Option<VectorConfig>,
    #[serde(default)]
    use_hnsw: bool,
    #[serde(default)]
    enable_bm25: bool,
    // Qdrant compatibility - sparse vectors (stored but not fully implemented)
    #[serde(default)]
    sparse_vectors: Option<serde_json::Value>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)] // Qdrant API compatibility - extra fields accepted but currently ignored
struct VectorConfig {
    size: usize,
    distance: Option<String>,
    // Qdrant compatibility - ignored fields
    #[serde(default)]
    on_disk: Option<bool>,
    #[serde(default)]
    hnsw_config: Option<serde_json::Value>,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
    #[serde(default)]
    multivector_config: Option<serde_json::Value>,
    #[serde(default)]
    datatype: Option<String>,
}

// Custom deserializer to handle both simple and named vector formats
fn deserialize_vectors_optional<'de, D>(deserializer: D) -> Result<Option<VectorConfig>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    
    let Some(value) = value else {
        return Ok(None);
    };
    
    // Try simple format first: {"size": 1536, "distance": "Cosine"}
    if let Ok(config) = serde_json::from_value::<VectorConfig>(value.clone()) {
        return Ok(Some(config));
    }
    
    // Try named vectors format: {"": {"size": 1536, ...}} or {"vector_name": {"size": 1536, ...}}
    if let Ok(named) = serde_json::from_value::<HashMap<String, VectorConfig>>(value.clone()) {
        // Use the first vector config (default or named)
        if let Some(config) = named.into_values().next() {
            return Ok(Some(config));
        }
    }
    
    Err(serde::de::Error::custom("Invalid vectors configuration: expected either {\"size\": N, \"distance\": \"...\"} or {\"name\": {\"size\": N, ...}}"))
}

// Note: Using serde_json::json! for flexible responses instead of these structs
#[allow(dead_code)]
#[derive(Serialize)]
struct CollectionInfo {
    name: String,
    vectors: VectorConfigResponse,
    points_count: usize,
}

#[allow(dead_code)]
#[derive(Serialize)]
struct VectorConfigResponse {
    size: usize,
    distance: String,
}

#[derive(Deserialize)]
struct UpsertPointsRequest {
    points: Vec<PointRequest>,
}

/// Parsed sparse vector data
struct ParsedSparseVector {
    /// Name of the sparse vector (e.g., "keywords")
    name: String,
    /// Indices of non-zero elements
    indices: Vec<u32>,
    /// Values at those indices
    values: Vec<f32>,
}

/// Parsed vector data - can be single, multi, or sparse
struct ParsedVector {
    /// First/primary vector (for backwards compatibility)
    primary: Vec<f32>,
    /// Full multivector data if this was a multivector input
    multivector: Option<Vec<Vec<f32>>>,
    /// Sparse vectors (named)
    sparse_vectors: Vec<ParsedSparseVector>,
}

#[derive(Deserialize)]
struct PointRequest {
    id: serde_json::Value,
    /// Vector is optional when using similarity schema (auto-embedding mode)
    #[serde(default, deserialize_with = "deserialize_vector_optional")]
    vector: Option<ParsedVector>,
    payload: Option<serde_json::Value>,
}

// Custom deserializer for optional vector (for similarity schema auto-embedding)
// Supports multiple formats (Qdrant compatibility):
//   Simple:      [0.1, 0.2, 0.3]
//   Multivector: [[0.1, 0.2], [0.3, 0.4]] -> stores full multivector for MaxSim search
//   Named:       {"vector_name": [0.1, 0.2]}
//   Sparse:      {"keywords": {"indices": [...], "values": [...]}}
fn deserialize_vector_optional<'de, D>(deserializer: D) -> Result<Option<ParsedVector>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    
    let Some(value) = value else {
        return Ok(None);
    };
    
    // Reuse the same parsing logic
    fn parse_simple_vector(arr: &[serde_json::Value]) -> Result<Vec<f32>, String> {
        arr.iter()
            .map(|v| v.as_f64().map(|f| f as f32).ok_or_else(|| "expected f32".to_string()))
            .collect()
    }
    
    fn parse_multivector(arr: &[serde_json::Value]) -> Result<Vec<Vec<f32>>, String> {
        arr.iter()
            .map(|sub| {
                if let serde_json::Value::Array(sub_arr) = sub {
                    parse_simple_vector(sub_arr)
                } else {
                    Err("expected array of arrays for multivector".to_string())
                }
            })
            .collect()
    }
    
    match &value {
        serde_json::Value::Array(arr) if !arr.is_empty() => {
            match arr.first() {
                Some(serde_json::Value::Number(_)) => {
                    let primary = parse_simple_vector(arr).map_err(serde::de::Error::custom)?;
                    Ok(Some(ParsedVector { primary, multivector: None, sparse_vectors: Vec::new() }))
                }
                Some(serde_json::Value::Array(_)) => {
                    let multivec = parse_multivector(arr).map_err(serde::de::Error::custom)?;
                    let primary = multivec.first().cloned().unwrap_or_default();
                    Ok(Some(ParsedVector { primary, multivector: Some(multivec), sparse_vectors: Vec::new() }))
                }
                _ => Err(serde::de::Error::custom("invalid vector format"))
            }
        }
        serde_json::Value::Array(_) => Ok(None), // Empty array treated as no vector
        serde_json::Value::Object(obj) => {
            let mut sparse_vectors = Vec::new();
            let mut primary = vec![0.0];
            let mut multivector = None;
            
            for (name, vec_value) in obj.iter() {
                match vec_value {
                    serde_json::Value::Array(arr) if !arr.is_empty() => {
                        match arr.first() {
                            Some(serde_json::Value::Number(_)) => {
                                primary = parse_simple_vector(arr).map_err(serde::de::Error::custom)?;
                            }
                            Some(serde_json::Value::Array(_)) => {
                                let multivec = parse_multivector(arr).map_err(serde::de::Error::custom)?;
                                primary = multivec.first().cloned().unwrap_or_default();
                                multivector = Some(multivec);
                            }
                            _ => {}
                        }
                    }
                    serde_json::Value::Object(sparse_obj) => {
                        if let (Some(indices_arr), Some(values_arr)) = (
                            sparse_obj.get("indices").and_then(|i| i.as_array()),
                            sparse_obj.get("values").and_then(|v| v.as_array())
                        ) {
                            let indices: Vec<u32> = indices_arr.iter()
                                .filter_map(|i| i.as_u64().map(|n| n as u32))
                                .collect();
                            let values: Vec<f32> = values_arr.iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect();
                            
                            if !indices.is_empty() && !values.is_empty() {
                                sparse_vectors.push(ParsedSparseVector {
                                    name: name.clone(),
                                    indices,
                                    values,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
            
            Ok(Some(ParsedVector { primary, multivector, sparse_vectors }))
        }
        serde_json::Value::Null => Ok(None),
        _ => Err(serde::de::Error::custom("vector must be an array, object, or null")),
    }
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Option<Vec<f32>>,
    text: Option<String>,
    #[serde(alias = "top")]
    limit: Option<usize>,
    filter: Option<serde_json::Value>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default)]
    with_vector: Option<bool>,
    #[serde(default)]
    score_threshold: Option<f32>,
    #[serde(default)]
    offset: Option<usize>,
}

#[allow(dead_code)]
#[derive(Serialize)]
struct SearchResult {
    id: serde_json::Value,
    score: f32,
    payload: Option<serde_json::Value>,
}

pub struct RestApi;

impl RestApi {
    pub async fn start(
        storage: Arc<StorageManager>,
        port: u16,
    ) -> std::io::Result<()> {
        Self::start_with_static_dir(storage, port, STATIC_DIR).await
    }
    
    pub async fn start_with_static_dir(
        storage: Arc<StorageManager>,
        port: u16,
        static_dir: &str,
    ) -> std::io::Result<()> {
        let static_folder = static_dir.to_string();
        
        HttpServer::new(move || {
            let cors = Cors::default()
                .allow_any_origin()
                .allow_any_method()
                .allow_any_header()
                .max_age(3600);

            let mut app = App::new()
                .wrap(cors)
                .app_data(web::Data::new(storage.clone()))
                // Service endpoints (Qdrant-compatible)
                .route("/", web::get().to(root_info))
                .route("/healthz", web::get().to(health_check))
                .route("/livez", web::get().to(livez_check))
                .route("/readyz", web::get().to(readyz_check))
                .route("/metrics", web::get().to(metrics_endpoint))
                // Collection endpoints
                .route("/collections", web::get().to(list_collections))
                .route("/collections/{name}", web::get().to(get_collection))
                .route("/collections/{name}", web::put().to(create_collection))
                .route("/collections/{name}", web::delete().to(delete_collection))
                .route("/collections/{name}/points", web::put().to(upsert_points))
                .route("/collections/{name}/points/scroll", web::post().to(scroll_points))
                .route("/collections/{name}/points/delete", web::post().to(delete_points_by_filter))
                .route("/collections/{name}/points/search", web::post().to(search_points))
                .route("/collections/{name}/points/query", web::post().to(query_points))
                .route("/collections/{name}/points/{id}", web::get().to(get_point))
                .route("/collections/{name}/points/{id}", web::delete().to(delete_point))
                .route("/collections/{name}/exists", web::get().to(collection_exists))
                // Qdrant compatibility - additional endpoints
                .route("/aliases", web::get().to(list_aliases))
                .route("/collections/aliases", web::post().to(update_aliases))
                .route("/collections/{name}/aliases", web::get().to(list_collection_aliases))
                .route("/cluster", web::get().to(cluster_info))
                .route("/collections/{name}/cluster", web::get().to(collection_cluster_info))
                .route("/telemetry", web::get().to(telemetry_info))
                // Points batch operations
                .route("/collections/{name}/points", web::post().to(get_points_by_ids))
                .route("/collections/{name}/points/count", web::post().to(count_points))
                .route("/collections/{name}/points/payload", web::post().to(set_payload))
                .route("/collections/{name}/points/payload", web::put().to(overwrite_payload))
                .route("/collections/{name}/points/payload/delete", web::post().to(delete_payload))
                .route("/collections/{name}/points/payload/clear", web::post().to(clear_payload))
                .route("/collections/{name}/points/vectors", web::put().to(update_vectors))
                .route("/collections/{name}/points/vectors/delete", web::post().to(delete_vectors))
                .route("/collections/{name}/points/batch", web::post().to(batch_update))
                .route("/collections/{name}/points/search/batch", web::post().to(batch_search))
                .route("/collections/{name}/points/search/groups", web::post().to(search_groups))
                .route("/collections/{name}/points/query/batch", web::post().to(batch_query))
                .route("/collections/{name}/points/query/groups", web::post().to(query_groups))
                .route("/collections/{name}/points/discover", web::post().to(discover_points))
                .route("/collections/{name}/points/discover/batch", web::post().to(discover_batch))
                .route("/collections/{name}/facet", web::post().to(facet_counts))
                // Index endpoints
                .route("/collections/{name}/index", web::put().to(create_field_index))
                .route("/collections/{name}/index/{field_name}", web::delete().to(delete_field_index))
                // Recommend endpoint
                .route("/collections/{name}/points/recommend", web::post().to(recommend_points))
                // Snapshot endpoints (stubs for UI compatibility)
                .route("/collections/{name}/snapshots", web::get().to(list_snapshots))
                .route("/collections/{name}/snapshots", web::post().to(create_snapshot))
                .route("/collections/{name}/snapshots/upload", web::post().to(upload_snapshot))
                .route("/collections/{name}/snapshots/recover", web::put().to(recover_snapshot))
                .route("/collections/{name}/snapshots/{snapshot_name}", web::get().to(get_snapshot))
                .route("/collections/{name}/snapshots/{snapshot_name}", web::delete().to(delete_snapshot))
                // Full storage snapshots
                .route("/snapshots", web::get().to(list_all_snapshots))
                .route("/snapshots", web::post().to(create_full_snapshot))
                .route("/snapshots/{snapshot_name}", web::get().to(get_full_snapshot))
                .route("/snapshots/{snapshot_name}", web::delete().to(delete_full_snapshot))
                // Collection update endpoint
                .route("/collections/{name}", web::patch().to(update_collection))
                // Issues endpoints
                .route("/issues", web::get().to(get_issues))
                .route("/issues", web::delete().to(clear_issues));
            
            // Serve web UI dashboard if static folder exists
            let static_path = Path::new(&static_folder);
            if static_path.exists() && static_path.is_dir() {
                app = app.service(
                    Files::new(DASHBOARD_PATH, static_folder.clone())
                        .index_file("index.html")
                        .use_last_modified(true)
                );
            }
            
            app
        })
        .bind(("0.0.0.0", port))?
        .run()
        .await
    }
}

async fn root_info() -> ActixResult<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "title": "vectx - vector search engine",
        "version": "0.2.1",
        "commit": ""
    })))
}

async fn health_check() -> ActixResult<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "title": "vectx",
        "version": "0.2.1"
    })))
}

/// Kubernetes liveness probe
async fn livez_check() -> ActixResult<HttpResponse> {
    Ok(HttpResponse::Ok()
        .content_type("text/plain")
        .body("healthz check passed"))
}

/// Kubernetes readiness probe  
async fn readyz_check() -> ActixResult<HttpResponse> {
    Ok(HttpResponse::Ok()
        .content_type("text/plain")
        .body("healthz check passed"))
}

/// Prometheus metrics endpoint
async fn metrics_endpoint(
    storage: web::Data<Arc<StorageManager>>,
) -> ActixResult<HttpResponse> {
    let collections = storage.list_collections();
    let collections_count = collections.len();
    
    // Count total points across all collections
    let mut total_points = 0u64;
    for name in &collections {
        if let Some(collection) = storage.get_collection(name) {
            total_points += collection.count() as u64;
        }
    }
    
    let metrics = format!(
        "# HELP app_info information about vectx server\n\
         # TYPE app_info gauge\n\
         app_info{{name=\"vectx\",version=\"{}\"}} 1\n\
         # HELP cluster_enabled is cluster support enabled\n\
         # TYPE cluster_enabled gauge\n\
         cluster_enabled 0\n\
         # HELP collections_total number of collections\n\
         # TYPE collections_total gauge\n\
         collections_total {}\n\
         # HELP points_total total number of points across all collections\n\
         # TYPE points_total gauge\n\
         points_total {}\n",
        env!("CARGO_PKG_VERSION"),
        collections_count,
        total_points
    );
    
    Ok(HttpResponse::Ok()
        .content_type("text/plain")
        .body(metrics))
}

async fn list_collections(
    storage: web::Data<Arc<StorageManager>>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let collection_names = storage.list_collections();
    
    // Format to match Qdrant's response structure (only name, no config)
    let collections: Vec<serde_json::Value> = collection_names.into_iter()
        .map(|name| serde_json::json!({ "name": name }))
        .collect();
    
    Ok(qdrant_response(serde_json::json!({
        "collections": collections
    }), start_time))
}

async fn get_collection(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    if let Some(collection) = storage.get_collection(&name) {
        let distance_str = format!("{:?}", collection.distance());
        let vector_dim = collection.vector_dim();
        let points_count = collection.count();
        
        // Format to match Qdrant's full response structure
        Ok(qdrant_response(serde_json::json!({
            "status": "green",
            "optimizer_status": "ok",
            "vectors_count": points_count,
            "indexed_vectors_count": points_count,
            "points_count": points_count,
            "segments_count": 1,
            "config": {
                "params": {
                    "vectors": {
                        "size": vector_dim,
                        "distance": distance_str
                    },
                    "shard_number": 1,
                    "replication_factor": 1,
                    "write_consistency_factor": 1,
                    "on_disk_payload": true
                },
                "hnsw_config": {
                    "m": 16,
                    "ef_construct": 100,
                    "full_scan_threshold": 10000,
                    "max_indexing_threads": 0,
                    "on_disk": false
                },
                "optimizer_config": {
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000,
                    "default_segment_number": 0,
                    "indexing_threshold": 10000,
                    "flush_interval_sec": 5,
                    "max_segment_size": null,
                    "memmap_threshold": null,
                    "max_optimization_threads": null
                },
                "wal_config": {
                    "wal_capacity_mb": 32,
                    "wal_segments_ahead": 0,
                    "wal_retain_closed": 1
                },
                "quantization_config": null
            },
            "payload_schema": {}
        }), start_time))
    } else {
        Ok(qdrant_not_found("Collection not found", start_time))
    }
}

async fn create_collection(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<CreateCollectionRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    // Handle sparse-only collections (Qdrant compatibility)
    // For sparse-only collections, we create with a default vector dimension
    let (vector_dim, distance) = if let Some(ref vectors) = req.vectors {
        let dist = match vectors.distance.as_deref() {
            Some("Cosine") | Some("cosine") => Distance::Cosine,
            Some("Euclidean") | Some("euclidean") => Distance::Euclidean,
            Some("Dot") | Some("dot") => Distance::Dot,
            _ => Distance::Cosine,
        };
        (vectors.size, dist)
    } else if req.sparse_vectors.is_some() {
        // Sparse-only collection - use BM25 with default text dimension
        (0, Distance::Cosine)
    } else {
        return Ok(qdrant_error("'vectors' configuration is required. Clients must provide embedding vectors.", start_time));
    };

    let config = CollectionConfig {
        name: name.clone(),
        vector_dim,
        distance,
        use_hnsw: req.use_hnsw,
        // Enable BM25 for sparse collections
        enable_bm25: req.enable_bm25 || req.sparse_vectors.is_some(),
    };

    match storage.create_collection(config) {
        Ok(_) => Ok(qdrant_response(true, start_time)),
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

async fn delete_collection(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    match storage.delete_collection(&name) {
        Ok(true) => Ok(qdrant_response(true, start_time)),
        Ok(false) => Ok(qdrant_not_found("Collection not found", start_time)),
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

async fn upsert_points(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<UpsertPointsRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };
    
    let points: Result<Vec<Point>, &str> = req.points.iter().map(|point_req| {
        let id = match &point_req.id {
            serde_json::Value::String(s) => PointId::String(s.clone()),
            serde_json::Value::Number(n) => {
                if let Some(u) = n.as_u64() {
                    PointId::Integer(u)
                } else {
                    return Err("Invalid point ID");
                }
            }
            _ => return Err("Invalid point ID"),
        };

        // Vector is required - clients must provide embeddings
        let mut point = match &point_req.vector {
            Some(parsed_vector) => {
                if let Some(ref multivec_data) = parsed_vector.multivector {
                    // Create MultiVector and Point with multivector
                    match MultiVector::new(multivec_data.clone()) {
                        Ok(mv) => Point::new_multi(id, mv, point_req.payload.clone()),
                        Err(_) => {
                            let vector = Vector::new(parsed_vector.primary.clone());
                            Point::new(id, vector, point_req.payload.clone())
                        }
                    }
                } else {
                    // Simple dense vector
                    let vector = Vector::new(parsed_vector.primary.clone());
                    Point::new(id, vector, point_req.payload.clone())
                }
            }
            None => {
                // No vector provided - use empty vector (for payload-only points)
                let vector = Vector::new(vec![]);
                Point::new(id, vector, point_req.payload.clone())
            }
        };
        
        // Add sparse vectors if present
        if let Some(ref parsed_vector) = point_req.vector {
            for sparse in &parsed_vector.sparse_vectors {
                let sparse_vec = vectx_core::SparseVector::new(
                    sparse.indices.clone(),
                    sparse.values.clone()
                );
                point.add_sparse_vector(sparse.name.clone(), sparse_vec);
            }
        }
        
        Ok(point)
    }).collect();

    match points {
        Ok(points_vec) => {
            if points_vec.len() > 1 {
                const PREWARM_THRESHOLD: usize = 1000;
                let should_prewarm = points_vec.len() >= PREWARM_THRESHOLD;
                
                let result = if should_prewarm {
                    collection.batch_upsert_with_prewarm(points_vec, true)
                } else {
                    collection.batch_upsert(points_vec)
                };
                
                if let Err(e) = result {
                    return Ok(qdrant_error(&e.to_string(), start_time));
                }
            } else if let Some(point) = points_vec.first() {
                if let Err(e) = collection.upsert(point.clone()) {
                    return Ok(qdrant_error(&e.to_string(), start_time));
                }
            }
        }
        Err(e) => {
            return Ok(qdrant_error(e, start_time));
        }
    }

    let operation_id = collection.next_operation_id();
    Ok(qdrant_response(serde_json::json!({
        "operation_id": operation_id,
        "status": "acknowledged"
    }), start_time))
}

async fn search_points(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<SearchRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    let limit = req.limit.unwrap_or(10);
    let with_payload = req.with_payload.unwrap_or(true);
    let with_vector = req.with_vector.unwrap_or(false);
    let score_threshold = req.score_threshold;
    let offset = req.offset.unwrap_or(0);

    if let Some(text) = &req.text {
        let results = collection.search_text(text, limit + offset);
        let search_results: Vec<serde_json::Value> = results
            .into_iter()
            .skip(offset)
            .filter(|(_, score)| score_threshold.map(|t| *score >= t).unwrap_or(true))
            .filter_map(|(doc_id, score)| {
                collection.get(&doc_id).map(|point| {
                    let mut result = serde_json::json!({
                        "id": point_id_to_json(&point.id),
                        "version": point.version,
                        "score": score,
                    });
                    if with_payload {
                        result["payload"] = point.payload.clone().unwrap_or(serde_json::Value::Null);
                    }
                    if with_vector {
                        result["vector"] = serde_json::json!(point.vector.as_slice());
                    }
                    result
                })
            })
            .collect();

        return Ok(qdrant_response(search_results, start_time));
    }

    if let Some(vector_data) = &req.vector {
        let query_vector = Vector::new(vector_data.clone());
        
        let filter: Option<Box<dyn Filter>> = req.filter.as_ref().and_then(|f| {
            parse_filter(f).map(|cond| Box::new(PayloadFilter::new(cond)) as Box<dyn Filter>)
        });

        let results = if let Some(f) = filter.as_deref() {
            collection.search(&query_vector, limit + offset, Some(f))
        } else {
            collection.search(&query_vector, limit + offset, None)
        };

        let search_results: Vec<serde_json::Value> = results
            .into_iter()
            .skip(offset)
            .filter(|(_, score)| score_threshold.map(|t| *score >= t).unwrap_or(true))
            .map(|(point, score)| {
                let mut result = serde_json::json!({
                    "id": point_id_to_json(&point.id),
                    "version": point.version,
                    "score": score,
                });
                if with_payload {
                    result["payload"] = point.payload.clone().unwrap_or(serde_json::Value::Null);
                }
                if with_vector {
                    result["vector"] = serde_json::json!(point.vector.as_slice());
                }
                result
            })
            .collect();

        return Ok(qdrant_response(search_results, start_time));
    }

    Ok(qdrant_error("Either 'vector' or 'text' must be provided", start_time))
}

/// Convert PointId to JSON value
fn point_id_to_json(id: &vectx_core::PointId) -> serde_json::Value {
    match id {
        vectx_core::PointId::String(s) => serde_json::Value::String(s.clone()),
        vectx_core::PointId::Integer(i) => serde_json::Value::Number((*i).into()),
        vectx_core::PointId::Uuid(u) => serde_json::Value::String(u.to_string()),
    }
}

/// Prefetch query for hybrid search
#[derive(Deserialize, Clone)]
struct PrefetchQuery {
    /// Query vector or sparse vector
    query: serde_json::Value,
    #[serde(default)]
    using: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

/// Query request for Qdrant's universal query API
/// Supports both single vectors and multivectors (ColBERT-style MaxSim)
#[derive(Deserialize)]
struct QueryRequest {
    /// Query vector - can be single [f32], multi [[f32]], or fusion object {"fusion": "rrf"}
    query: serde_json::Value,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default)]
    with_vector: Option<bool>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
    /// Prefetch queries for hybrid search
    #[serde(default)]
    prefetch: Option<Vec<PrefetchQuery>>,
    /// Which named vector to use
    #[serde(default)]
    using: Option<String>,
}

/// Query points using Qdrant's universal query API
/// Supports multivector queries with MaxSim scoring, prefetch, and fusion
async fn query_points(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<QueryRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    let limit = req.limit.unwrap_or(10);
    let with_payload = req.with_payload.unwrap_or(true);
    let with_vector = req.with_vector.unwrap_or(false);
    
    // Check if this is a fusion query with prefetch
    let is_fusion = req.query.as_object()
        .and_then(|o| o.get("fusion"))
        .is_some();
    
    let results = if is_fusion && req.prefetch.is_some() {
        // Handle hybrid search with prefetch and fusion
        match execute_fusion_query(&collection, &req, limit) {
            Ok(r) => r,
            Err(e) => return Ok(qdrant_error(&e, start_time)),
        }
    } else {
        // Parse filter if provided
        let filter: Option<Box<dyn Filter>> = req.filter.as_ref().and_then(|f| {
            parse_filter(f).map(|cond| Box::new(PayloadFilter::new(cond)) as Box<dyn Filter>)
        });
        
        // Get the "using" parameter for named/sparse vector queries
        let using = req.using.as_deref();
        
        // Determine query type: point ID, single vector, sparse, or multivector
        match execute_simple_query(&collection, &req.query, limit, filter.as_deref(), using) {
            Ok(r) => r,
            Err(e) => return Ok(qdrant_error(&e, start_time)),
        }
    };
    
    // Format results
    let search_results: Vec<serde_json::Value> = results
        .into_iter()
        .map(|(point, score)| {
            let mut result = serde_json::json!({
                "id": point_id_to_json(&point.id),
                "version": point.version,
                "score": score,
            });
            
            if with_payload {
                result["payload"] = point.payload.clone().unwrap_or(serde_json::Value::Null);
            }
            
            if with_vector {
                result["vector"] = serde_json::json!(point.vector.as_slice());
                if let Some(mv) = &point.multivector {
                    result["multivector"] = serde_json::json!(mv.vectors());
                }
            }
            
            result
        })
        .collect();

    Ok(qdrant_response(serde_json::json!({
        "points": search_results
    }), start_time))
}

/// Execute a fusion query with prefetch (RRF - Reciprocal Rank Fusion)
fn execute_fusion_query(
    collection: &Arc<Collection>,
    req: &QueryRequest,
    limit: usize,
) -> Result<Vec<(Point, f32)>, String> {
    use std::collections::HashMap;
    
    let prefetch = req.prefetch.as_ref().ok_or("Fusion requires prefetch")?;
    
    // Execute each prefetch query and collect ranked results
    let mut all_results: Vec<Vec<(Point, f32)>> = Vec::new();
    
    for pf in prefetch {
        let pf_limit = pf.limit.unwrap_or(20);
        let filter: Option<Box<dyn Filter>> = pf.filter.as_ref().and_then(|f| {
            parse_filter(f).map(|cond| Box::new(PayloadFilter::new(cond)) as Box<dyn Filter>)
        });
        
        // Parse the prefetch query, using the "using" parameter for named/sparse vectors
        let using = pf.using.as_deref();
        let pf_results = parse_and_search(collection, &pf.query, pf_limit, filter.as_deref(), using)?;
        all_results.push(pf_results);
    }
    
    // Apply RRF (Reciprocal Rank Fusion)
    // RRF score = sum(1 / (k + rank_i)) for each result set
    // Qdrant uses k=1 for consistent scoring
    const K: f32 = 1.0;
    
    let mut rrf_scores: HashMap<String, (Point, f32)> = HashMap::new();
    
    for result_set in &all_results {
        for (rank, (point, _original_score)) in result_set.iter().enumerate() {
            let rrf_contribution = 1.0 / (K + rank as f32 + 1.0);
            let point_id = point.id.to_string();
            
            rrf_scores
                .entry(point_id)
                .and_modify(|(_, score)| *score += rrf_contribution)
                .or_insert_with(|| (point.clone(), rrf_contribution));
        }
    }
    
    // Sort by RRF score descending
    let mut fused: Vec<(Point, f32)> = rrf_scores.into_values().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);
    
    Ok(fused)
}

/// Parse a query value and execute search
fn parse_and_search(
    collection: &Arc<Collection>,
    query: &serde_json::Value,
    limit: usize,
    filter: Option<&dyn Filter>,
    using: Option<&str>,
) -> Result<Vec<(Point, f32)>, String> {
    match query {
        // Sparse vector format: {"indices": [...], "values": [...]}
        serde_json::Value::Object(obj) if obj.contains_key("indices") && obj.contains_key("values") => {
            let indices = obj.get("indices")
                .and_then(|i| i.as_array())
                .ok_or("Invalid sparse vector: missing indices")?;
            let values = obj.get("values")
                .and_then(|v| v.as_array())
                .ok_or("Invalid sparse vector: missing values")?;
            
            let indices_vec: Vec<u32> = indices.iter()
                .filter_map(|i| i.as_u64().map(|n| n as u32))
                .collect();
            let values_vec: Vec<f32> = values.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            
            if indices_vec.is_empty() || values_vec.is_empty() {
                return Ok(Vec::new());
            }
            
            // Use the "using" parameter as the sparse vector name, default to "default"
            let vector_name = using.unwrap_or("default");
            let query_sparse = vectx_core::SparseVector::new(indices_vec, values_vec);
            
            // Perform sparse dot product search
            Ok(collection.search_sparse(&query_sparse, vector_name, limit, filter))
        }
        // Array: single vector or multivector
        serde_json::Value::Array(arr) if !arr.is_empty() => {
            match arr.first() {
                // Multivector: [[0.1, 0.2], [0.3, 0.4]]
                Some(serde_json::Value::Array(_)) => {
                    let multivec_data: Result<Vec<Vec<f32>>, _> = arr.iter()
                        .map(|sub| {
                            if let serde_json::Value::Array(sub_arr) = sub {
                                sub_arr.iter()
                                    .map(|v| v.as_f64().map(|f| f as f32).ok_or("expected f32"))
                                    .collect::<Result<Vec<f32>, _>>()
                            } else {
                                Err("expected array")
                            }
                        })
                        .collect();
                    
                    let data = multivec_data.map_err(|e| format!("Invalid multivector: {}", e))?;
                    let query_mv = MultiVector::new(data).map_err(|e| format!("Invalid multivector: {}", e))?;
                    Ok(collection.search_multivector(&query_mv, limit, filter))
                }
                // Single vector: [0.1, 0.2, 0.3]
                Some(serde_json::Value::Number(_)) => {
                    let vector_data: Result<Vec<f32>, _> = arr.iter()
                        .map(|v| v.as_f64().map(|f| f as f32).ok_or("expected f32"))
                        .collect();
                    
                    let data = vector_data.map_err(|e| format!("Invalid vector: {}", e))?;
                    let query_vector = Vector::new(data);
                    Ok(collection.search(&query_vector, limit, filter))
                }
                _ => Err("Invalid query format".to_string())
            }
        }
        _ => Err("Invalid query format".to_string())
    }
}

/// Execute a simple (non-fusion) query
fn execute_simple_query(
    collection: &Arc<Collection>,
    query: &serde_json::Value,
    limit: usize,
    filter: Option<&dyn Filter>,
    using: Option<&str>,
) -> Result<Vec<(Point, f32)>, String> {
    match query {
        // Query by point ID (nearest to existing point)
        serde_json::Value::Number(n) => {
            let point_id_str = if let Some(id) = n.as_u64() {
                id.to_string()
            } else if let Some(id) = n.as_i64() {
                id.to_string()
            } else {
                return Err("Invalid point ID format".to_string());
            };
            
            // Get the point by ID and use its vector for search
            if let Some(source_point) = collection.get(&point_id_str) {
                let query_vector = source_point.vector.clone();
                let mut search_results = collection.search(&query_vector, limit + 1, filter);
                // Remove the source point from results
                search_results.retain(|(p, _)| p.id.to_string() != point_id_str);
                search_results.truncate(limit);
                Ok(search_results)
            } else {
                Err(format!("Point with ID '{}' not found", point_id_str))
            }
        }
        // Query by string point ID
        serde_json::Value::String(s) => {
            if let Some(source_point) = collection.get(s) {
                let query_vector = source_point.vector.clone();
                let mut search_results = collection.search(&query_vector, limit + 1, filter);
                // Remove the source point from results
                search_results.retain(|(p, _)| p.id.to_string() != *s);
                search_results.truncate(limit);
                Ok(search_results)
            } else {
                Err(format!("Point with ID '{}' not found", s))
            }
        }
        // Arrays and sparse vectors
        _ => parse_and_search(collection, query, limit, filter, using)
    }
}

/// Parse Qdrant-style filter format (must/should/must_not)
fn parse_filter(filter_json: &serde_json::Value) -> Option<FilterCondition> {
    if let Some(obj) = filter_json.as_object() {
        let mut all_conditions: Vec<FilterCondition> = Vec::new();
        
        // Qdrant-style filter with must (AND conditions)
        if let Some(must) = obj.get("must") {
            if let Some(arr) = must.as_array() {
                let must_conditions: Vec<FilterCondition> = arr.iter()
                    .filter_map(parse_field_condition)
                    .collect();
                if !must_conditions.is_empty() {
                    // Multiple must conditions are AND'd together
                    if must_conditions.len() == 1 {
                        all_conditions.push(must_conditions.into_iter().next().unwrap());
                    } else {
                        all_conditions.push(FilterCondition::And(must_conditions));
                    }
                }
            }
        }
        
        // Qdrant-style filter with should (OR conditions)
        if let Some(should) = obj.get("should") {
            if let Some(arr) = should.as_array() {
                let should_conditions: Vec<FilterCondition> = arr.iter()
                    .filter_map(parse_field_condition)
                    .collect();
                if !should_conditions.is_empty() {
                    // Multiple should conditions are OR'd together
                    if should_conditions.len() == 1 {
                        all_conditions.push(should_conditions.into_iter().next().unwrap());
                    } else {
                        all_conditions.push(FilterCondition::Or(should_conditions));
                    }
                }
            }
        }
        
        // Qdrant-style filter with must_not (negated conditions)
        if let Some(must_not) = obj.get("must_not") {
            if let Some(arr) = must_not.as_array() {
                for cond in arr {
                    if let Some(fc) = parse_field_condition(cond) {
                        all_conditions.push(FilterCondition::Not(Box::new(fc)));
                    }
                }
            }
        }
        
        // If we collected any conditions, combine them
        if !all_conditions.is_empty() {
            return if all_conditions.len() == 1 {
                Some(all_conditions.into_iter().next().unwrap())
            } else {
                Some(FilterCondition::And(all_conditions))
            };
        }
        
        // Legacy simple format: { field, value, operator }
        if let Some(field) = obj.get("field").and_then(|v| v.as_str()) {
            if let Some(value) = obj.get("value") {
                if let Some(op) = obj.get("operator").and_then(|v| v.as_str()) {
                    match op {
                        "eq" => return Some(FilterCondition::Equals { field: field.to_string(), value: value.clone() }),
                        "ne" => return Some(FilterCondition::NotEquals { field: field.to_string(), value: value.clone() }),
                        "gt" => return value.as_f64().map(|v| FilterCondition::GreaterThan { field: field.to_string(), value: v }),
                        "lt" => return value.as_f64().map(|v| FilterCondition::LessThan { field: field.to_string(), value: v }),
                        "gte" => return value.as_f64().map(|v| FilterCondition::GreaterEqual { field: field.to_string(), value: v }),
                        "lte" => return value.as_f64().map(|v| FilterCondition::LessEqual { field: field.to_string(), value: v }),
                        _ => {}
                    }
                }
            }
        }
        
        // Direct field condition (Qdrant format): { "key": "field", "match": { "value": x } }
        if let Some(fc) = parse_field_condition(filter_json) {
            return Some(fc);
        }
    }
    None
}

/// Parse a single Qdrant field condition: { "key": "field", "match": { "value": x } }
fn parse_field_condition(cond: &serde_json::Value) -> Option<FilterCondition> {
    let obj = cond.as_object()?;
    let key = obj.get("key")?.as_str()?;
    
    // Match condition: { "match": { "value": x } }
    if let Some(match_obj) = obj.get("match").and_then(|m| m.as_object()) {
        if let Some(value) = match_obj.get("value") {
            return Some(FilterCondition::Equals { 
                field: key.to_string(), 
                value: value.clone() 
            });
        }
        // Match any: { "match": { "any": [x, y, z] } }
        if let Some(any_arr) = match_obj.get("any").and_then(|a| a.as_array()) {
            if let Some(first) = any_arr.first() {
                return Some(FilterCondition::Equals { 
                    field: key.to_string(), 
                    value: first.clone() 
                });
            }
        }
        // Match text: { "match": { "text": "value" } }
        if let Some(text) = match_obj.get("text") {
            return Some(FilterCondition::Equals { 
                field: key.to_string(), 
                value: text.clone() 
            });
        }
    }
    
    // Range condition: { "range": { "gt": x, "lt": y } }
    if let Some(range_obj) = obj.get("range").and_then(|r| r.as_object()) {
        if let Some(gt) = range_obj.get("gt").and_then(|v| v.as_f64()) {
            return Some(FilterCondition::GreaterThan { field: key.to_string(), value: gt });
        }
        if let Some(gte) = range_obj.get("gte").and_then(|v| v.as_f64()) {
            return Some(FilterCondition::GreaterEqual { field: key.to_string(), value: gte });
        }
        if let Some(lt) = range_obj.get("lt").and_then(|v| v.as_f64()) {
            return Some(FilterCondition::LessThan { field: key.to_string(), value: lt });
        }
        if let Some(lte) = range_obj.get("lte").and_then(|v| v.as_f64()) {
            return Some(FilterCondition::LessEqual { field: key.to_string(), value: lte });
        }
    }
    
    None
}

/// Check if a point matches a Qdrant-style filter
fn matches_filter(point: &Point, filter: &serde_json::Value) -> bool {
    let obj = match filter.as_object() {
        Some(o) => o,
        None => return true, // No valid filter, match all
    };
    
    // Handle "must" conditions (AND logic)
    if let Some(must) = obj.get("must").and_then(|m| m.as_array()) {
        for cond in must {
            if !matches_condition(point, cond) {
                return false; // All must conditions must match
            }
        }
    }
    
    // Handle "should" conditions (OR logic)
    if let Some(should) = obj.get("should").and_then(|s| s.as_array()) {
        if !should.is_empty() {
            let any_match = should.iter().any(|cond| matches_condition(point, cond));
            if !any_match {
                return false; // At least one should condition must match
            }
        }
    }
    
    // Handle "must_not" conditions (NOT logic)
    if let Some(must_not) = obj.get("must_not").and_then(|m| m.as_array()) {
        for cond in must_not {
            if matches_condition(point, cond) {
                return false; // No must_not condition should match
            }
        }
    }
    
    true
}

/// Check if a point matches a single condition
fn matches_condition(point: &Point, cond: &serde_json::Value) -> bool {
    let obj = match cond.as_object() {
        Some(o) => o,
        None => return false,
    };
    
    // Handle "has_id" condition - filter by point IDs
    if let Some(ids) = obj.get("has_id").and_then(|h| h.as_array()) {
        let point_id_str = point.id.to_string();
        return ids.iter().any(|id| {
            match id {
                serde_json::Value::Number(n) => n.to_string() == point_id_str,
                serde_json::Value::String(s) => s == &point_id_str,
                _ => false,
            }
        });
    }
    
    // Handle "nested" filter - match conditions within same array element
    if let Some(nested_obj) = obj.get("nested").and_then(|n| n.as_object()) {
        let nested_key = match nested_obj.get("key").and_then(|k| k.as_str()) {
            Some(k) => k,
            None => return false,
        };
        let nested_filter = match nested_obj.get("filter") {
            Some(f) => f,
            None => return false,
        };
        
        // Get the array field from payload
        let array_value = match &point.payload {
            Some(payload) => payload.get(nested_key).and_then(|v| v.as_array()),
            None => None,
        };
        
        // Check if any array element matches ALL conditions in the nested filter
        if let Some(arr) = array_value {
            return arr.iter().any(|element| {
                matches_nested_element(element, nested_filter)
            });
        }
        return false;
    }
    
    // Get the field key
    let key = match obj.get("key").and_then(|k| k.as_str()) {
        Some(k) => k,
        None => {
            // Handle nested filter (recursive)
            if obj.contains_key("must") || obj.contains_key("should") || obj.contains_key("must_not") {
                return matches_filter(point, cond);
            }
            return false;
        }
    };
    
    // Get the payload value for this key (supports nested paths like "diet[].food")
    let payload_value = get_nested_value(&point.payload, key);
    
    // Handle "match" condition
    if let Some(match_obj) = obj.get("match").and_then(|m| m.as_object()) {
        // Match exact value
        if let Some(expected) = match_obj.get("value") {
            return match &payload_value {
                Some(actual) => values_equal(actual, expected),
                None => false,
            };
        }
        
        // Match any of values
        if let Some(any_arr) = match_obj.get("any").and_then(|a| a.as_array()) {
            return match &payload_value {
                Some(actual) => any_arr.iter().any(|expected| values_equal(actual, expected)),
                None => false,
            };
        }
        
        // Match text (words OR - any word in query matches)
        if let Some(text) = match_obj.get("text").and_then(|t| t.as_str()) {
            return match &payload_value {
                Some(serde_json::Value::String(s)) => {
                    let s_lower = s.to_lowercase();
                    // Split query into words and check if any word matches
                    text.split_whitespace()
                        .any(|word| s_lower.contains(&word.to_lowercase()))
                }
                _ => false,
            };
        }
    }
    
    // Handle "range" condition
    if let Some(range_obj) = obj.get("range").and_then(|r| r.as_object()) {
        let actual_num = match &payload_value {
            Some(serde_json::Value::Number(n)) => n.as_f64(),
            _ => None,
        };
        
        if let Some(actual) = actual_num {
            if let Some(gt) = range_obj.get("gt").and_then(|v| v.as_f64()) {
                if actual <= gt { return false; }
            }
            if let Some(gte) = range_obj.get("gte").and_then(|v| v.as_f64()) {
                if actual < gte { return false; }
            }
            if let Some(lt) = range_obj.get("lt").and_then(|v| v.as_f64()) {
                if actual >= lt { return false; }
            }
            if let Some(lte) = range_obj.get("lte").and_then(|v| v.as_f64()) {
                if actual > lte { return false; }
            }
            return true;
        }
        return false;
    }
    
    false
}

/// Get a value from a nested path like "diet[].food" or "nested.field"
fn get_nested_value(payload: &Option<serde_json::Value>, path: &str) -> Option<serde_json::Value> {
    let payload = payload.as_ref()?;
    
    // Handle array notation like "diet[].food"
    if path.contains("[]") {
        let parts: Vec<&str> = path.split("[]").collect();
        if parts.len() >= 2 {
            let array_key = parts[0];
            let field_path = parts[1].trim_start_matches('.');
            
            if let Some(arr) = payload.get(array_key).and_then(|v| v.as_array()) {
                // Collect all matching values from array elements
                let values: Vec<serde_json::Value> = arr.iter()
                    .filter_map(|element| {
                        if field_path.is_empty() {
                            Some(element.clone())
                        } else {
                            element.get(field_path).cloned()
                        }
                    })
                    .collect();
                
                if !values.is_empty() {
                    return Some(serde_json::Value::Array(values));
                }
            }
        }
        return None;
    }
    
    // Handle dot notation like "nested.field"
    if path.contains('.') {
        let mut current = payload;
        for part in path.split('.') {
            current = current.get(part)?;
        }
        return Some(current.clone());
    }
    
    // Simple key lookup
    payload.get(path).cloned()
}

/// Check if an array element matches a nested filter
fn matches_nested_element(element: &serde_json::Value, filter: &serde_json::Value) -> bool {
    let filter_obj = match filter.as_object() {
        Some(o) => o,
        None => return false,
    };
    
    // Handle "must" conditions within the element
    if let Some(must) = filter_obj.get("must").and_then(|m| m.as_array()) {
        for cond in must {
            if !matches_element_condition(element, cond) {
                return false;
            }
        }
    }
    
    // Handle "should" conditions
    if let Some(should) = filter_obj.get("should").and_then(|s| s.as_array()) {
        if !should.is_empty() {
            let any_match = should.iter().any(|cond| matches_element_condition(element, cond));
            if !any_match {
                return false;
            }
        }
    }
    
    // Handle "must_not" conditions
    if let Some(must_not) = filter_obj.get("must_not").and_then(|m| m.as_array()) {
        for cond in must_not {
            if matches_element_condition(element, cond) {
                return false;
            }
        }
    }
    
    true
}

/// Check if an array element matches a single condition
fn matches_element_condition(element: &serde_json::Value, cond: &serde_json::Value) -> bool {
    let obj = match cond.as_object() {
        Some(o) => o,
        None => return false,
    };
    
    let key = match obj.get("key").and_then(|k| k.as_str()) {
        Some(k) => k,
        None => return false,
    };
    
    let element_value = element.get(key);
    
    if let Some(match_obj) = obj.get("match").and_then(|m| m.as_object()) {
        if let Some(expected) = match_obj.get("value") {
            return match element_value {
                Some(actual) => values_equal(actual, expected),
                None => false,
            };
        }
    }
    
    false
}

/// Compare two JSON values for equality
fn values_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::String(s1), serde_json::Value::String(s2)) => s1 == s2,
        (serde_json::Value::Number(n1), serde_json::Value::Number(n2)) => {
            n1.as_f64() == n2.as_f64()
        }
        (serde_json::Value::Bool(b1), serde_json::Value::Bool(b2)) => b1 == b2,
        (serde_json::Value::Array(arr), val) | (val, serde_json::Value::Array(arr)) => {
            // Check if val is in array
            arr.iter().any(|item| values_equal(item, val))
        }
        _ => a == b,
    }
}

#[derive(Deserialize)]
struct ScrollRequest {
    limit: Option<usize>,
    offset: Option<serde_json::Value>,
    with_payload: Option<bool>,
    with_vector: Option<bool>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

async fn scroll_points(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<ScrollRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let collection_name = path.into_inner();
    
    let collection = match storage.get_collection(&collection_name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };
    
    let limit = req.limit.unwrap_or(10);
    let with_payload = req.with_payload.unwrap_or(true);
    let with_vector = req.with_vector.unwrap_or(false);
    
    // Get offset as integer if provided
    let offset_id: Option<i64> = req.offset.as_ref().and_then(|v| {
        match v {
            serde_json::Value::Number(n) => n.as_i64(),
            serde_json::Value::String(s) => s.parse().ok(),
            _ => None,
        }
    });
    
    // Get all points and sort by ID for consistent pagination
    let all_points = collection.get_all_points();
    
    // Apply filter if provided
    let filtered_points: Vec<_> = if let Some(filter_json) = &req.filter {
        all_points.iter()
            .filter(|p| matches_filter(p, filter_json))
            .collect()
    } else {
        all_points.iter().collect()
    };
    
    let mut points_with_ids: Vec<_> = filtered_points.iter()
        .map(|p| {
            let id_num: i64 = match &p.id {
                vectx_core::PointId::Integer(i) => *i as i64,
                vectx_core::PointId::String(s) => s.parse::<i64>().unwrap_or(0),
                vectx_core::PointId::Uuid(_) => 0,
            };
            (id_num, *p)
        })
        .collect();
    
    points_with_ids.sort_by_key(|(id, _)| *id);
    
    // Apply offset
    let start_idx = if let Some(offset) = offset_id {
        points_with_ids.iter().position(|(id, _)| *id > offset).unwrap_or(points_with_ids.len())
    } else {
        0
    };
    
    // Get page of results
    let page: Vec<_> = points_with_ids.iter()
        .skip(start_idx)
        .take(limit)
        .collect();
    
    // Determine next offset
    let next_offset = if page.len() == limit && start_idx + limit < points_with_ids.len() {
        page.last().map(|(id, _)| serde_json::json!(*id))
    } else {
        None
    };
    
    // Format results
    let results: Vec<serde_json::Value> = page.iter().map(|(_, point)| {
        let mut obj = serde_json::json!({
            "id": point_id_to_json(&point.id),
            "version": point.version,
        });
        
        if with_payload {
            obj["payload"] = point.payload.clone().unwrap_or(serde_json::json!({}));
        }
        if with_vector {
            obj["vector"] = serde_json::json!(point.vector.as_slice());
        }
        
        obj
    }).collect();
    
    Ok(qdrant_response(serde_json::json!({
        "points": results,
        "next_page_offset": next_offset
    }), start_time))
}

async fn get_point(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<(String, String)>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let (collection_name, point_id) = path.into_inner();
    
    let collection = match storage.get_collection(&collection_name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    match collection.get(&point_id) {
        Some(point) => {
            // Build response with optional multivector
            let mut result = serde_json::json!({
                "id": point_id_to_json(&point.id),
                "version": point.version,
                "vector": point.vector.as_slice(),
                "payload": point.payload.clone().unwrap_or(serde_json::Value::Null),
            });
            
            // Add multivector if present
            if let Some(mv) = &point.multivector {
                result["multivector"] = serde_json::json!(mv.vectors());
            }
            
            Ok(qdrant_response(result, start_time))
        }
        None => Ok(qdrant_not_found("Point not found", start_time)),
    }
}

async fn delete_point(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<(String, String)>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let (collection_name, point_id) = path.into_inner();
    
    let collection = match storage.get_collection(&collection_name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    match collection.delete(&point_id) {
        Ok(true) => {
            let operation_id = collection.next_operation_id();
            Ok(qdrant_response(serde_json::json!({
                "operation_id": operation_id,
                "status": "acknowledged"
            }), start_time))
        }
        Ok(false) => Ok(qdrant_not_found("Point not found", start_time)),
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

#[derive(Deserialize)]
struct DeletePointsRequest {
    filter: Option<DeleteFilter>,
    points: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
struct DeleteFilter {
    must: Option<Vec<FilterMust>>,
}

#[derive(Deserialize)]
struct FilterMust {
    key: String,
    #[serde(rename = "match")]
    match_value: MatchValue,
}

#[derive(Deserialize)]
struct MatchValue {
    value: serde_json::Value,
}

async fn delete_points_by_filter(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<DeletePointsRequest>,
) -> ActixResult<HttpResponse> {
    let collection_name = path.into_inner();
    
    let start_time = Instant::now();
    let collection = match storage.get_collection(&collection_name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };
    
    let mut _deleted_count = 0;
    
    // Handle filter-based deletion
    if let Some(filter) = &req.filter {
        if let Some(must_conditions) = &filter.must {
            // Get the field and value to match
            if let Some(condition) = must_conditions.first() {
                let field_key = &condition.key;
                let match_value = &condition.match_value.value;
                
                // Get all points and filter by payload
                let all_points = collection.get_all_points();
                let mut points_to_delete = Vec::new();
                
                for point in all_points {
                    if let Some(payload) = &point.payload {
                        if let Some(field_value) = payload.get(field_key) {
                            if field_value == match_value {
                                points_to_delete.push(point.id.clone());
                            }
                        }
                    }
                }
                
                // Delete matching points
                for point_id in points_to_delete {
                    let id_str = match &point_id {
                        vectx_core::PointId::String(s) => s.clone(),
                        vectx_core::PointId::Integer(i) => i.to_string(),
                        vectx_core::PointId::Uuid(u) => u.to_string(),
                    };
                    if collection.delete(&id_str).is_ok() {
                        _deleted_count += 1;
                    }
                }
            }
        }
    }
    
    // Handle point ID-based deletion
    if let Some(point_ids) = &req.points {
        for point_id in point_ids {
            let id_str = match point_id {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if collection.delete(&id_str).is_ok() {
                _deleted_count += 1;
            }
        }
    }
    
    let operation_id = collection.next_operation_id();
    Ok(qdrant_response(serde_json::json!({
        "operation_id": operation_id,
        "status": "acknowledged"
    }), start_time))
}

async fn collection_exists(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    let exists = storage.collection_exists(&name);
    
    Ok(qdrant_response(serde_json::json!({
        "exists": exists
    }), start_time))
}

// Qdrant compatibility endpoints

async fn list_aliases(
    storage: web::Data<Arc<StorageManager>>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let aliases: Vec<serde_json::Value> = storage.list_aliases()
        .into_iter()
        .map(|(alias, collection)| serde_json::json!({
            "alias_name": alias,
            "collection_name": collection
        }))
        .collect();
    Ok(qdrant_response(serde_json::json!({
        "aliases": aliases
    }), start_time))
}

async fn cluster_info() -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    // vectX runs as single node - return disabled status like Qdrant single-node
    Ok(qdrant_response(serde_json::json!({
        "status": "disabled"
    }), start_time))
}

async fn telemetry_info() -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    Ok(qdrant_response(serde_json::json!({
        "id": "vectx-single-node",
        "app": {
            "name": "vectx",
            "version": "0.2.1"
        }
    }), start_time))
}

// Snapshot endpoints

async fn list_snapshots(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let collection_name = path.into_inner();
    
    match storage.list_collection_snapshots(&collection_name) {
        Ok(snapshots) => Ok(qdrant_response(snapshots, start_time)),
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

async fn list_all_snapshots(
    storage: web::Data<Arc<StorageManager>>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    match storage.list_all_snapshots() {
        Ok(snapshots) => Ok(qdrant_response(snapshots, start_time)),
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

/// Create full storage snapshot
async fn create_full_snapshot(
    storage: web::Data<Arc<StorageManager>>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    
    // Create snapshots for all collections
    let collections = storage.list_collections();
    let mut created_snapshots = Vec::new();
    
    for collection_name in collections {
        match storage.create_collection_snapshot(&collection_name) {
            Ok(snapshot) => created_snapshots.push(snapshot),
            Err(e) => {
                return Ok(qdrant_error(&format!("Failed to snapshot {}: {}", collection_name, e), start_time));
            }
        }
    }
    
    // Return metadata about full snapshot
    let snapshot_name = format!("full-snapshot-{}.snapshot", Utc::now().format("%Y-%m-%d-%H-%M-%S"));
    
    Ok(qdrant_response(serde_json::json!({
        "name": snapshot_name,
        "creation_time": Utc::now().to_rfc3339(),
        "size": 0,
        "collections": created_snapshots.len()
    }), start_time))
}

/// Get (download) full snapshot
async fn get_full_snapshot(
    _path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    Ok(qdrant_error("Full storage snapshot download not yet implemented", start_time))
}

/// Delete full snapshot
async fn delete_full_snapshot(
    _path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    // For now, just acknowledge
    Ok(qdrant_response(true, start_time))
}

/// Update collection parameters
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - update params not yet implemented
struct UpdateCollectionRequest {
    #[serde(default)]
    optimizers_config: Option<serde_json::Value>,
    #[serde(default)]
    params: Option<serde_json::Value>,
    #[serde(default)]
    hnsw_config: Option<serde_json::Value>,
    #[serde(default)]
    vectors: Option<serde_json::Value>,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

async fn update_collection(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    _req: web::Json<UpdateCollectionRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    if !storage.collection_exists(&name) {
        return Ok(qdrant_not_found("Collection not found", start_time));
    }
    
    // Collection update acknowledged (parameters update not yet fully implemented)
    Ok(qdrant_response(true, start_time))
}

/// Get issues/performance suggestions
async fn get_issues() -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    Ok(qdrant_response(serde_json::json!({
        "issues": []
    }), start_time))
}

/// Clear all reported issues
async fn clear_issues() -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    Ok(qdrant_response(true, start_time))
}

async fn create_snapshot(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let collection_name = path.into_inner();
    
    // Check if collection exists
    if !storage.collection_exists(&collection_name) {
        return Ok(qdrant_not_found(&format!("Collection '{}' not found", collection_name), start_time));
    }
    
    match storage.create_collection_snapshot(&collection_name) {
        Ok(snapshot) => Ok(qdrant_response(snapshot, start_time)),
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - priority/checksum currently informational
struct RecoverSnapshotRequest {
    location: String,
    #[serde(default)]
    priority: Option<String>,
    #[serde(default)]
    checksum: Option<String>,
}

async fn recover_snapshot(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<RecoverSnapshotRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let collection_name = path.into_inner();
    let location = &req.location;
    
    // Helper to build response with collection info
    fn build_recovery_result(collection: &vectx_core::Collection) -> serde_json::Value {
        let points_count = collection.count();
        let vector_dim = collection.vector_dim();
        
        if points_count == 0 {
            serde_json::json!({
                "recovered": true,
                "collection": collection.name(),
                "vector_dim": vector_dim,
                "points_count": 0,
                "note": "Collection created with config only. If this was a Qdrant snapshot, points must be migrated separately using the scroll API."
            })
        } else {
            serde_json::json!({
                "recovered": true,
                "collection": collection.name(),
                "vector_dim": vector_dim,
                "points_count": points_count
            })
        }
    }
    
    // Check if it's a URL or local file reference
    if location.starts_with("http://") || location.starts_with("https://") {
        // Remote URL recovery
        match storage.recover_from_url(
            &collection_name,
            location,
            req.checksum.as_deref(),
        ).await {
            Ok(collection) => Ok(qdrant_response(build_recovery_result(&collection), start_time)),
            Err(e) => Ok(qdrant_error(&format!("Failed to recover from URL: {}", e), start_time)),
        }
    } else if location.starts_with("file://") {
        // Local file recovery - extract snapshot name from path
        let snapshot_name = location
            .trim_start_matches("file://")
            .rsplit('/')
            .next()
            .unwrap_or(location);
        
        match storage.recover_from_snapshot(&collection_name, snapshot_name) {
            Ok(collection) => Ok(qdrant_response(build_recovery_result(&collection), start_time)),
            Err(e) => Ok(qdrant_error(&format!("Failed to recover from snapshot: {}", e), start_time)),
        }
    } else {
        // Assume it's a snapshot name directly
        match storage.recover_from_snapshot(&collection_name, location) {
            Ok(collection) => Ok(qdrant_response(build_recovery_result(&collection), start_time)),
            Err(e) => Ok(qdrant_error(&format!("Failed to recover from snapshot: {}", e), start_time)),
        }
    }
}

async fn get_snapshot(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<(String, String)>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let (collection_name, snapshot_name) = path.into_inner();
    
    if let Some(snapshot_path) = storage.get_snapshot_path(&collection_name, &snapshot_name) {
        // Return the snapshot file for download
        match std::fs::read(&snapshot_path) {
            Ok(data) => {
                Ok(HttpResponse::Ok()
                    .content_type("application/octet-stream")
                    .insert_header(("Content-Disposition", format!("attachment; filename=\"{}\"", snapshot_name)))
                    .body(data))
            }
            Err(e) => Ok(qdrant_error(&format!("Failed to read snapshot file: {}", e), start_time)),
        }
    } else {
        Ok(qdrant_not_found(&format!("Snapshot '{}' not found in collection '{}'", snapshot_name, collection_name), start_time))
    }
}

async fn delete_snapshot(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<(String, String)>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let (collection_name, snapshot_name) = path.into_inner();
    
    match storage.delete_collection_snapshot(&collection_name, &snapshot_name) {
        Ok(true) => Ok(qdrant_response(true, start_time)),
        Ok(false) => Ok(qdrant_not_found(&format!("Snapshot '{}' not found in collection '{}'", snapshot_name, collection_name), start_time)),
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

async fn upload_snapshot(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    mut payload: Multipart,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let collection_name = path.into_inner();
    
    let mut snapshot_data: Option<Vec<u8>> = None;
    let mut filename: Option<String> = None;
    
    // Process multipart form data
    while let Some(item) = payload.next().await {
        let mut field = match item {
            Ok(f) => f,
            Err(e) => {
                return Ok(qdrant_error(&format!("Failed to parse multipart: {}", e), start_time));
            }
        };
        
        let content_disposition = match field.content_disposition() {
            Some(cd) => cd,
            None => continue,
        };
        let field_name = content_disposition.get_name().unwrap_or("");
        
        if field_name == "snapshot" {
            // Get filename
            filename = content_disposition.get_filename().map(|s: &str| s.to_string());
            
            // Read file data
            let mut data = Vec::new();
            while let Some(chunk) = field.next().await {
                match chunk {
                    Ok(bytes) => data.extend_from_slice(&bytes),
                    Err(e) => {
                        return Ok(qdrant_error(&format!("Failed to read file data: {}", e), start_time));
                    }
                }
            }
            snapshot_data = Some(data);
        }
    }
    
    // Validate we got the snapshot file
    let data = match snapshot_data {
        Some(d) => d,
        None => {
            return Ok(qdrant_error("No snapshot file provided in multipart form", start_time));
        }
    };
    
    // Save and restore the snapshot
    match storage.upload_and_restore_snapshot(&collection_name, &data, filename.as_deref()) {
        Ok(collection) => Ok(qdrant_response(serde_json::json!({
            "collection": collection_name,
            "points_count": collection.count()
        }), start_time)),
        Err(e) => Ok(qdrant_error(&format!("Failed to restore snapshot: {}", e), start_time)),
    }
}

// ============ Additional Qdrant-compatible endpoints ============

/// Update aliases (stub - aliases not yet implemented)
/// Update aliases request
#[derive(Deserialize)]
struct UpdateAliasesRequest {
    actions: Vec<serde_json::Value>,
}

async fn update_aliases(
    storage: web::Data<Arc<StorageManager>>,
    req: web::Json<UpdateAliasesRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    
    for action in &req.actions {
        if let Some(obj) = action.as_object() {
            // Create alias: { "create_alias": { "alias_name": "x", "collection_name": "y" } }
            if let Some(create) = obj.get("create_alias").and_then(|c| c.as_object()) {
                if let (Some(alias), Some(collection)) = (
                    create.get("alias_name").and_then(|v| v.as_str()),
                    create.get("collection_name").and_then(|v| v.as_str())
                ) {
                    let _ = storage.create_alias(alias, collection);
                }
            }
            
            // Delete alias: { "delete_alias": { "alias_name": "x" } }
            if let Some(delete) = obj.get("delete_alias").and_then(|d| d.as_object()) {
                if let Some(alias) = delete.get("alias_name").and_then(|v| v.as_str()) {
                    let _ = storage.delete_alias(alias);
                }
            }
            
            // Rename alias: { "rename_alias": { "old_alias_name": "x", "new_alias_name": "y" } }
            if let Some(rename) = obj.get("rename_alias").and_then(|r| r.as_object()) {
                if let (Some(old_alias), Some(new_alias)) = (
                    rename.get("old_alias_name").and_then(|v| v.as_str()),
                    rename.get("new_alias_name").and_then(|v| v.as_str())
                ) {
                    let _ = storage.rename_alias(old_alias, new_alias);
                }
            }
        }
    }
    
    Ok(qdrant_response(true, start_time))
}

/// List collection aliases
async fn list_collection_aliases(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let collection_name = path.into_inner();
    let aliases: Vec<serde_json::Value> = storage.list_collection_aliases(&collection_name)
        .into_iter()
        .map(|alias| serde_json::json!({
            "alias_name": alias,
            "collection_name": collection_name
        }))
        .collect();
    Ok(qdrant_response(serde_json::json!({
        "aliases": aliases
    }), start_time))
}

/// Collection cluster info
async fn collection_cluster_info(
    path: web::Path<String>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let collection_name = path.into_inner();
    Ok(qdrant_response(serde_json::json!({
        "peer_id": 0,
        "shard_count": 1,
        "local_shards": [{
            "shard_id": 0,
            "points_count": 0,
            "state": "Active"
        }],
        "remote_shards": [],
        "shard_transfers": [],
        "collection_name": collection_name
    }), start_time))
}

/// Get multiple points by IDs
#[derive(Deserialize)]
struct GetPointsRequest {
    ids: Vec<serde_json::Value>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default)]
    with_vector: Option<bool>,
}

async fn get_points_by_ids(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<GetPointsRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    let with_payload = req.with_payload.unwrap_or(true);
    let with_vector = req.with_vector.unwrap_or(false);
    
    let mut points = Vec::new();
    for id_value in &req.ids {
        let id_str = match id_value {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            _ => continue,
        };
        
        if let Some(point) = collection.get(&id_str) {
            let mut result = serde_json::json!({
                "id": id_value,
                "version": point.version
            });
            if with_payload {
                result["payload"] = point.payload.clone().unwrap_or(serde_json::Value::Null);
            }
            if with_vector {
                result["vector"] = serde_json::json!(point.vector.as_slice());
            }
            points.push(result);
        }
    }

    Ok(qdrant_response(points, start_time))
}

/// Count points in collection
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - filter/exact accepted but not yet applied
struct CountRequest {
    #[serde(default)]
    filter: Option<serde_json::Value>,
    #[serde(default)]
    exact: Option<bool>,
}

async fn count_points(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    _req: web::Json<CountRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    Ok(qdrant_response(serde_json::json!({
        "count": collection.count()
    }), start_time))
}

/// Set payload on points
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - filter accepted but not yet applied
struct SetPayloadRequest {
    payload: serde_json::Value,
    #[serde(default)]
    points: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

async fn set_payload(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<SetPayloadRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    let mut updated_count = 0;

    // If specific points are provided, update only those
    if let Some(point_ids) = &req.points {
        for id_value in point_ids {
            let id_str = match id_value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if collection.set_payload(&id_str, req.payload.clone()).unwrap_or(false) {
                updated_count += 1;
            }
        }
    } else {
        // Update all points (or filtered points)
        let all_points = collection.get_all_points();
        for point in all_points {
            let id_str = point.id.to_string();
            if collection.set_payload(&id_str, req.payload.clone()).unwrap_or(false) {
                updated_count += 1;
            }
        }
    }

    Ok(qdrant_response(serde_json::json!({
        "operation_id": updated_count,
        "status": "acknowledged"
    }), start_time))
}

/// Overwrite payload on points (replace entire payload)
async fn overwrite_payload(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<SetPayloadRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    let mut updated_count = 0;

    if let Some(point_ids) = &req.points {
        for id_value in point_ids {
            let id_str = match id_value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if collection.overwrite_payload(&id_str, req.payload.clone()).unwrap_or(false) {
                updated_count += 1;
            }
        }
    } else {
        let all_points = collection.get_all_points();
        for point in all_points {
            let id_str = point.id.to_string();
            if collection.overwrite_payload(&id_str, req.payload.clone()).unwrap_or(false) {
                updated_count += 1;
            }
        }
    }

    Ok(qdrant_response(serde_json::json!({
        "operation_id": updated_count,
        "status": "acknowledged"
    }), start_time))
}

/// Delete payload fields from points
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - filter accepted but not yet applied
struct DeletePayloadRequest {
    keys: Vec<String>,
    #[serde(default)]
    points: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

async fn delete_payload(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<DeletePayloadRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    let mut updated_count = 0;

    if let Some(point_ids) = &req.points {
        for id_value in point_ids {
            let id_str = match id_value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if collection.delete_payload_keys(&id_str, &req.keys).unwrap_or(false) {
                updated_count += 1;
            }
        }
    } else {
        let all_points = collection.get_all_points();
        for point in all_points {
            let id_str = point.id.to_string();
            if collection.delete_payload_keys(&id_str, &req.keys).unwrap_or(false) {
                updated_count += 1;
            }
        }
    }

    Ok(qdrant_response(serde_json::json!({
        "operation_id": updated_count,
        "status": "acknowledged"
    }), start_time))
}

/// Clear all payload from points
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - filter accepted but not yet applied
struct ClearPayloadRequest {
    #[serde(default)]
    points: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

async fn clear_payload(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<ClearPayloadRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    let mut updated_count = 0;

    if let Some(point_ids) = &req.points {
        for id_value in point_ids {
            let id_str = match id_value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if collection.clear_payload(&id_str).unwrap_or(false) {
                updated_count += 1;
            }
        }
    } else {
        let all_points = collection.get_all_points();
        for point in all_points {
            let id_str = point.id.to_string();
            if collection.clear_payload(&id_str).unwrap_or(false) {
                updated_count += 1;
            }
        }
    }

    Ok(qdrant_response(serde_json::json!({
        "operation_id": updated_count,
        "status": "acknowledged"
    }), start_time))
}

/// Update vectors on existing points
#[derive(Deserialize)]
struct UpdateVectorsRequest {
    /// List of point updates with id and vector
    points: Vec<UpdateVectorPoint>,
}

#[derive(Deserialize)]
struct UpdateVectorPoint {
    id: serde_json::Value,
    vector: serde_json::Value,
}

async fn update_vectors(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<UpdateVectorsRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    let mut updated_count = 0;

    for point_update in &req.points {
        let id_str = match &point_update.id {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            _ => continue,
        };

        // Parse vector - can be array or named vectors object
        let vector_data = match &point_update.vector {
            serde_json::Value::Array(arr) => {
                let vec: Result<Vec<f32>, _> = arr.iter()
                    .map(|v| v.as_f64().map(|f| f as f32).ok_or("expected f32"))
                    .collect();
                vec.ok()
            }
            serde_json::Value::Object(obj) => {
                // Named vectors - get first one
                if let Some((_, vec_val)) = obj.iter().next() {
                    if let Some(arr) = vec_val.as_array() {
                        let vec: Result<Vec<f32>, _> = arr.iter()
                            .map(|v| v.as_f64().map(|f| f as f32).ok_or("expected f32"))
                            .collect();
                        vec.ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(vec) = vector_data {
            let vector = Vector::new(vec);
            if collection.update_vector(&id_str, vector).unwrap_or(false) {
                updated_count += 1;
            }
        }
    }

    Ok(qdrant_response(serde_json::json!({
        "operation_id": updated_count,
        "status": "acknowledged"
    }), start_time))
}

/// Delete vectors from points
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - filter accepted but not yet applied
struct DeleteVectorsRequest {
    #[serde(default)]
    points: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    vectors: Vec<String>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

async fn delete_vectors(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<DeleteVectorsRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    let mut deleted_count = 0;

    // Note: In a full named-vectors implementation, this would delete specific named vectors
    // For now, if points are specified, we clear their vectors (effectively delete the point)
    if let Some(point_ids) = &req.points {
        for id_value in point_ids {
            let id_str = match id_value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => continue,
            };
            // Delete multivector if it was the target
            if req.vectors.iter().any(|v| v == "multivector" || v.is_empty())
                && collection.update_multivector(&id_str, None).unwrap_or(false) {
                    deleted_count += 1;
                }
        }
    }

    Ok(qdrant_response(serde_json::json!({
        "operation_id": deleted_count,
        "status": "acknowledged"
    }), start_time))
}

/// Batch update operations
#[derive(Deserialize)]
struct BatchUpdateRequest {
    operations: Vec<serde_json::Value>,
}

async fn batch_update(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<BatchUpdateRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    let mut results = Vec::new();

    for (idx, operation) in req.operations.iter().enumerate() {
        let op_result = process_batch_operation(&collection, operation);
        results.push(serde_json::json!({
            "operation_id": idx,
            "status": if op_result { "acknowledged" } else { "failed" }
        }));
    }

    Ok(qdrant_response(results, start_time))
}

/// Process a single batch operation
fn process_batch_operation(collection: &std::sync::Arc<vectx_core::Collection>, operation: &serde_json::Value) -> bool {
    let obj = match operation.as_object() {
        Some(o) => o,
        None => return false,
    };

    // Upsert operation
    if let Some(upsert) = obj.get("upsert") {
        if let Some(points) = upsert.get("points").and_then(|p| p.as_array()) {
            for point_json in points {
                if let Some(point) = parse_point_from_json(point_json) {
                    let _ = collection.upsert(point);
                }
            }
            return true;
        }
    }

    // Delete operation
    if let Some(delete) = obj.get("delete") {
        if let Some(points) = delete.get("points").and_then(|p| p.as_array()) {
            for id_val in points {
                let id_str = match id_val {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    _ => continue,
                };
                let _ = collection.delete(&id_str);
            }
            return true;
        }
    }

    // Set payload operation
    if let Some(set_payload) = obj.get("set_payload") {
        if let Some(payload) = set_payload.get("payload") {
            if let Some(points) = set_payload.get("points").and_then(|p| p.as_array()) {
                for id_val in points {
                    let id_str = match id_val {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Number(n) => n.to_string(),
                        _ => continue,
                    };
                    let _ = collection.set_payload(&id_str, payload.clone());
                }
                return true;
            }
        }
    }

    // Overwrite payload operation
    if let Some(overwrite_payload) = obj.get("overwrite_payload") {
        if let Some(payload) = overwrite_payload.get("payload") {
            if let Some(points) = overwrite_payload.get("points").and_then(|p| p.as_array()) {
                for id_val in points {
                    let id_str = match id_val {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Number(n) => n.to_string(),
                        _ => continue,
                    };
                    let _ = collection.overwrite_payload(&id_str, payload.clone());
                }
                return true;
            }
        }
    }

    // Delete payload operation
    if let Some(delete_payload) = obj.get("delete_payload") {
        if let Some(keys) = delete_payload.get("keys").and_then(|k| k.as_array()) {
            let key_strings: Vec<String> = keys.iter()
                .filter_map(|k| k.as_str().map(String::from))
                .collect();
            if let Some(points) = delete_payload.get("points").and_then(|p| p.as_array()) {
                for id_val in points {
                    let id_str = match id_val {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Number(n) => n.to_string(),
                        _ => continue,
                    };
                    let _ = collection.delete_payload_keys(&id_str, &key_strings);
                }
                return true;
            }
        }
    }

    // Clear payload operation
    if let Some(clear_payload) = obj.get("clear_payload") {
        if let Some(points) = clear_payload.get("points").and_then(|p| p.as_array()) {
            for id_val in points {
                let id_str = match id_val {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    _ => continue,
                };
                let _ = collection.clear_payload(&id_str);
            }
            return true;
        }
    }

    false
}

/// Parse a point from JSON
fn parse_point_from_json(json: &serde_json::Value) -> Option<Point> {
    let obj = json.as_object()?;
    
    let id = match obj.get("id")? {
        serde_json::Value::String(s) => vectx_core::PointId::String(s.clone()),
        serde_json::Value::Number(n) => {
            vectx_core::PointId::Integer(n.as_u64().unwrap_or(0))
        }
        _ => return None,
    };

    let vector_data = obj.get("vector")?;
    let vector = match vector_data {
        serde_json::Value::Array(arr) => {
            let vec: Result<Vec<f32>, _> = arr.iter()
                .map(|v| v.as_f64().map(|f| f as f32).ok_or("expected f32"))
                .collect();
            Vector::new(vec.ok()?)
        }
        _ => return None,
    };

    let payload = obj.get("payload").cloned();

    Some(Point::new(id, vector, payload))
}

/// Batch search
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - batch dispatch not yet implemented
struct BatchSearchRequest {
    searches: Vec<serde_json::Value>,
}

async fn batch_search(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    _req: web::Json<BatchSearchRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    if storage.get_collection(&name).is_none() {
        return Ok(qdrant_not_found("Collection not found", start_time));
    }

    Ok(qdrant_response(Vec::<serde_json::Value>::new(), start_time))
}

/// Search points grouped by a payload field
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - filter accepted but not yet applied
struct SearchGroupsRequest {
    vector: Vec<f32>,
    group_by: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    group_size: Option<usize>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default)]
    with_vector: Option<bool>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

async fn search_groups(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<SearchGroupsRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    let limit = req.limit.unwrap_or(5);
    let group_size = req.group_size.unwrap_or(3);
    let with_payload = req.with_payload.unwrap_or(true);
    let with_vector = req.with_vector.unwrap_or(false);
    let group_by = &req.group_by;
    
    let query_vector = Vector::new(req.vector.clone());
    let search_results = collection.search(&query_vector, limit * group_size * 2, None);
    
    // Group results by the group_by field
    let mut groups: std::collections::HashMap<String, Vec<serde_json::Value>> = std::collections::HashMap::new();
    
    for (point, score) in search_results {
        let group_key = point.payload
            .as_ref()
            .and_then(|p| p.get(group_by))
            .and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "unknown".to_string());
        
        let group = groups.entry(group_key).or_default();
        
        if group.len() < group_size {
            let mut hit = serde_json::json!({
                "id": point_id_to_json(&point.id),
                "score": score
            });
            
            if with_payload {
                hit["payload"] = point.payload.clone().unwrap_or(serde_json::Value::Null);
            }
            if with_vector {
                hit["vector"] = serde_json::json!(point.vector.as_slice());
            }
            
            group.push(hit);
        }
        
        if groups.len() >= limit && groups.values().all(|g| g.len() >= group_size) {
            break;
        }
    }
    
    let group_results: Vec<serde_json::Value> = groups
        .into_iter()
        .take(limit)
        .map(|(key, hits)| serde_json::json!({ "id": key, "hits": hits }))
        .collect();

    Ok(qdrant_response(serde_json::json!({
        "groups": group_results
    }), start_time))
}

/// Discover points using context pairs
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - context/filter accepted but discovery uses target only
struct DiscoverRequest {
    #[serde(default)]
    target: Option<serde_json::Value>,
    #[serde(default)]
    context: Option<Vec<ContextPair>>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default)]
    with_vector: Option<bool>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - context pairs accepted but not yet used
struct ContextPair {
    positive: serde_json::Value,
    negative: serde_json::Value,
}

async fn discover_points(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<DiscoverRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    let limit = req.limit.unwrap_or(10);
    let with_payload = req.with_payload.unwrap_or(true);
    let _with_vector = req.with_vector.unwrap_or(false);
    
    // Parse target vector or point ID
    let target_vector = if let Some(target) = &req.target {
        match target {
            serde_json::Value::Array(arr) => {
                let vec: Result<Vec<f32>, _> = arr.iter()
                    .map(|v| v.as_f64().map(|f| f as f32).ok_or("expected f32"))
                    .collect();
                vec.ok().map(Vector::new)
            }
            serde_json::Value::Number(n) => {
                let id = n.to_string();
                collection.get(&id).map(|p| p.vector.clone())
            }
            serde_json::Value::String(s) => {
                collection.get(s).map(|p| p.vector.clone())
            }
            _ => None,
        }
    } else {
        None
    };

    let query = match target_vector {
        Some(v) => v,
        None => {
            return Ok(qdrant_error("Target vector or point ID required", start_time));
        }
    };
    
    let results = collection.search(&query, limit, None);
    
    let scored_points: Vec<serde_json::Value> = results.into_iter().map(|(point, score)| {
        let mut result = serde_json::json!({
            "id": point_id_to_json(&point.id),
            "version": point.version,
            "score": score,
        });
        if with_payload {
            result["payload"] = point.payload.clone().unwrap_or(serde_json::Value::Null);
        }
        result
    }).collect();

    Ok(qdrant_response(scored_points, start_time))
}

/// Batch discover points
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - batch dispatch not yet implemented
struct DiscoverBatchRequest {
    searches: Vec<serde_json::Value>,
}

async fn discover_batch(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    _req: web::Json<DiscoverBatchRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    if storage.get_collection(&name).is_none() {
        return Ok(qdrant_not_found("Collection not found", start_time));
    }

    Ok(qdrant_response(Vec::<Vec<serde_json::Value>>::new(), start_time))
}

/// Facet counts - count points by unique payload values
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - filter/exact accepted but not yet applied
struct FacetRequest {
    key: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
    #[serde(default)]
    exact: Option<bool>,
}

async fn facet_counts(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<FacetRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    let limit = req.limit.unwrap_or(10);
    let key = &req.key;
    
    // Count occurrences of each value for the given key
    let all_points = collection.get_all_points();
    let mut value_counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    
    for point in all_points {
        if let Some(payload) = &point.payload {
            if let Some(value) = payload.get(key) {
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    _ => continue,
                };
                *value_counts.entry(value_str).or_insert(0) += 1;
            }
        }
    }
    
    // Sort by count and take top limit
    let mut counts: Vec<_> = value_counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    
    let hits: Vec<serde_json::Value> = counts.into_iter()
        .take(limit)
        .map(|(value, count)| serde_json::json!({
            "value": value,
            "count": count
        }))
        .collect();

    Ok(qdrant_response(serde_json::json!({
        "hits": hits
    }), start_time))
}

/// Batch query
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - batch dispatch not yet implemented
struct BatchQueryRequest {
    searches: Vec<serde_json::Value>,
}

async fn batch_query(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    _req: web::Json<BatchQueryRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    if storage.get_collection(&name).is_none() {
        return Ok(qdrant_not_found("Collection not found", start_time));
    }

    Ok(qdrant_response(Vec::<serde_json::Value>::new(), start_time))
}

/// Query points with grouping
#[derive(Deserialize)]
#[allow(dead_code)] // Qdrant API compatibility - filter accepted but not yet applied
struct QueryGroupsRequest {
    query: serde_json::Value,
    group_by: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    group_size: Option<usize>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default)]
    with_vector: Option<bool>,
    #[serde(default)]
    filter: Option<serde_json::Value>,
}

async fn query_groups(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<QueryGroupsRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    let limit = req.limit.unwrap_or(5);
    let group_size = req.group_size.unwrap_or(3);
    let with_payload = req.with_payload.unwrap_or(true);
    let with_vector = req.with_vector.unwrap_or(false);
    let group_by = &req.group_by;
    
    // Parse query vector
    let query_vector = match &req.query {
        serde_json::Value::Array(arr) => {
            let vec: Result<Vec<f32>, _> = arr.iter()
                .map(|v| v.as_f64().map(|f| f as f32).ok_or("expected f32"))
                .collect();
            match vec {
                Ok(v) => Vector::new(v),
                Err(_) => {
                    return Ok(qdrant_error("Invalid query vector", start_time));
                }
            }
        }
        _ => {
            return Ok(qdrant_error("Query must be a vector array", start_time));
        }
    };
    
    // Search for points
    let search_results = collection.search(&query_vector, limit * group_size * 2, None);
    
    // Group results by the group_by field
    let mut groups: std::collections::HashMap<String, Vec<serde_json::Value>> = std::collections::HashMap::new();
    
    for (point, score) in search_results {
        // Get group key from payload
        let group_key = point.payload
            .as_ref()
            .and_then(|p| p.get(group_by))
            .and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "unknown".to_string());
        
        let group = groups.entry(group_key).or_default();
        
        // Only add if group hasn't reached group_size
        if group.len() < group_size {
            let mut hit = serde_json::json!({
                "id": point_id_to_json(&point.id),
                "score": score
            });
            
            if with_payload {
                hit["payload"] = point.payload.clone().unwrap_or(serde_json::Value::Null);
            }
            if with_vector {
                hit["vector"] = serde_json::json!(point.vector.as_slice());
            }
            
            group.push(hit);
        }
        
        // Stop if we have enough groups
        if groups.len() >= limit && groups.values().all(|g| g.len() >= group_size) {
            break;
        }
    }
    
    // Format response
    let group_results: Vec<serde_json::Value> = groups
        .into_iter()
        .take(limit)
        .map(|(key, hits)| {
            serde_json::json!({
                "id": key,
                "hits": hits
            })
        })
        .collect();

    Ok(qdrant_response(serde_json::json!({
        "groups": group_results
    }), start_time))
}

/// Create field index
#[derive(Deserialize)]
struct CreateIndexRequest {
    field_name: String,
    #[serde(default)]
    field_schema: Option<serde_json::Value>,
}

async fn create_field_index(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<CreateIndexRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    // Parse field schema to determine index type
    let index_type = if let Some(schema) = &req.field_schema {
        match schema {
            serde_json::Value::String(s) => match s.as_str() {
                "keyword" => vectx_core::PayloadIndexType::Keyword,
                "integer" => vectx_core::PayloadIndexType::Integer,
                "float" => vectx_core::PayloadIndexType::Float,
                "bool" => vectx_core::PayloadIndexType::Bool,
                "geo" => vectx_core::PayloadIndexType::Geo,
                "text" => vectx_core::PayloadIndexType::Text,
                _ => vectx_core::PayloadIndexType::Keyword,
            }
            serde_json::Value::Object(obj) => {
                if let Some(type_val) = obj.get("type").and_then(|v| v.as_str()) {
                    match type_val {
                        "keyword" => vectx_core::PayloadIndexType::Keyword,
                        "integer" => vectx_core::PayloadIndexType::Integer,
                        "float" => vectx_core::PayloadIndexType::Float,
                        "bool" => vectx_core::PayloadIndexType::Bool,
                        "geo" => vectx_core::PayloadIndexType::Geo,
                        "text" => vectx_core::PayloadIndexType::Text,
                        _ => vectx_core::PayloadIndexType::Keyword,
                    }
                } else {
                    vectx_core::PayloadIndexType::Keyword
                }
            }
            _ => vectx_core::PayloadIndexType::Keyword,
        }
    } else {
        vectx_core::PayloadIndexType::Keyword
    };

    match collection.create_payload_index(&req.field_name, index_type) {
        Ok(_) => {
            let operation_id = collection.next_operation_id();
            Ok(qdrant_response(serde_json::json!({
                "operation_id": operation_id,
                "status": "acknowledged"
            }), start_time))
        }
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

/// Delete field index
async fn delete_field_index(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<(String, String)>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let (name, field_name) = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => return Ok(qdrant_not_found("Collection not found", start_time)),
    };

    match collection.delete_payload_index(&field_name) {
        Ok(_) => {
            let operation_id = collection.next_operation_id();
            Ok(qdrant_response(serde_json::json!({
                "operation_id": operation_id,
                "status": "acknowledged"
            }), start_time))
        }
        Err(e) => Ok(qdrant_error(&e.to_string(), start_time)),
    }
}

/// Recommend points based on positive/negative examples
/// Uses average vector strategy: query = 2*avg(positive) - avg(negative)
/// This creates a single query vector that moves toward positive examples
/// and away from negative examples, then performs one efficient search.
#[derive(Deserialize)]
struct RecommendRequest {
    #[serde(default)]
    positive: Vec<serde_json::Value>,
    #[serde(default)]
    negative: Vec<serde_json::Value>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default)]
    with_vector: Option<bool>,
    #[serde(default)]
    score_threshold: Option<f32>,
}

async fn recommend_points(
    storage: web::Data<Arc<StorageManager>>,
    path: web::Path<String>,
    req: web::Json<RecommendRequest>,
) -> ActixResult<HttpResponse> {
    let start_time = Instant::now();
    let name = path.into_inner();
    
    let collection = match storage.get_collection(&name) {
        Some(c) => c,
        None => {
            return Ok(qdrant_not_found("Collection not found", start_time));
        }
    };

    let limit = req.limit.unwrap_or(10);
    let with_payload = req.with_payload.unwrap_or(true);
    let with_vector = req.with_vector.unwrap_or(false);
    let score_threshold = req.score_threshold;
    
    // Collect point IDs to exclude from results
    let mut exclude_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    
    // Helper to parse point ID from JSON
    let parse_id = |id: &serde_json::Value| -> Option<String> {
        match id {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            _ => None,
        }
    };
    
    // Collect positive vectors and compute average
    let mut positive_vectors: Vec<Vec<f32>> = Vec::new();
    for pos_id in &req.positive {
        if let Some(id_str) = parse_id(pos_id) {
            exclude_ids.insert(id_str.clone());
            if let Some(point) = collection.get(&id_str) {
                positive_vectors.push(point.vector.as_slice().to_vec());
            }
        }
    }
    
    if positive_vectors.is_empty() {
        return Ok(qdrant_error("At least one valid positive example is required", start_time));
    }
    
    // Collect negative vectors and compute average
    let mut negative_vectors: Vec<Vec<f32>> = Vec::new();
    for neg_id in &req.negative {
        if let Some(id_str) = parse_id(neg_id) {
            exclude_ids.insert(id_str.clone());
            if let Some(point) = collection.get(&id_str) {
                negative_vectors.push(point.vector.as_slice().to_vec());
            }
        }
    }
    
    // Compute average of positive vectors
    let dim = positive_vectors[0].len();
    let mut avg_positive = vec![0.0f32; dim];
    for vec in &positive_vectors {
        for (i, &val) in vec.iter().enumerate() {
            if i < dim {
                avg_positive[i] += val;
            }
        }
    }
    let pos_count = positive_vectors.len() as f32;
    for val in &mut avg_positive {
        *val /= pos_count;
    }
    
    // Create query vector: if negatives exist, use formula 2*avg_pos - avg_neg
    // This moves the query toward positives and away from negatives
    let query_vector = if !negative_vectors.is_empty() {
        let mut avg_negative = vec![0.0f32; dim];
        for vec in &negative_vectors {
            for (i, &val) in vec.iter().enumerate() {
                if i < dim {
                    avg_negative[i] += val;
                }
            }
        }
        let neg_count = negative_vectors.len() as f32;
        for val in &mut avg_negative {
            *val /= neg_count;
        }
        
        // Combined query: 2*avg_pos - avg_neg
        avg_positive.iter()
            .zip(avg_negative.iter())
            .map(|(&pos, &neg)| 2.0 * pos - neg)
            .collect::<Vec<f32>>()
    } else {
        avg_positive
    };
    
    // Perform single search with combined query vector
    let query = Vector::new(query_vector);
    
    // Request more results to account for excluded IDs
    let search_limit = limit + exclude_ids.len();
    let search_results = collection.search(&query, search_limit, None);
    
    // Build results, excluding input point IDs
    let mut results = Vec::with_capacity(limit);
    for (point, score) in search_results {
        // Skip excluded points (the positive/negative examples)
        let point_id_str = point.id.to_string();
        if exclude_ids.contains(&point_id_str) {
            continue;
        }
        
        // Apply score threshold if provided
        if let Some(threshold) = score_threshold {
            if score < threshold {
                continue;
            }
        }
        
        let mut result = serde_json::json!({
            "id": point_id_to_json(&point.id),
            "version": point.version,
            "score": score
        });
        
        if with_payload {
            result["payload"] = point.payload.clone().unwrap_or(serde_json::Value::Null);
        }
        if with_vector {
            result["vector"] = serde_json::json!(point.vector.as_slice());
        }
        
        results.push(result);
        
        if results.len() >= limit {
            break;
        }
    }

    Ok(qdrant_response(results, start_time))
}
