use std::sync::Arc;
use std::time::Instant;
use tonic::{Request, Response, Status};
use vectx_storage::StorageManager;
use vectx_core::{Point, PointId, Vector, Distance as CoreDistance};

pub mod vectx {
    tonic::include_proto!("vectx");
}

use vectx::*;

// ============================================================================
// Qdrant Service (Health Check)
// ============================================================================

pub struct QdrantService;

#[tonic::async_trait]
impl vectx::qdrant_server::Qdrant for QdrantService {
    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckReply>, Status> {
        Ok(Response::new(HealthCheckReply {
            title: "vectx".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            commit: None,
        }))
    }
}

// ============================================================================
// Collections Service
// ============================================================================

pub struct CollectionsService {
    storage: Arc<StorageManager>,
}

impl CollectionsService {
    pub fn new(storage: Arc<StorageManager>) -> Self {
        Self { storage }
    }
}

#[tonic::async_trait]
impl vectx::collections_server::Collections for CollectionsService {
    async fn get(
        &self,
        request: Request<GetCollectionInfoRequest>,
    ) -> Result<Response<GetCollectionInfoResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;
        
        let points_count = collection.count() as u64;
        let vector_dim = collection.vector_dim() as u64;
        let distance = match collection.distance() {
            CoreDistance::Cosine => Distance::Cosine,
            CoreDistance::Euclidean => Distance::Euclid,
            CoreDistance::Dot => Distance::Dot,
        };

        let result = CollectionInfo {
            status: CollectionStatus::Green as i32,
            optimizer_status: Some(OptimizerStatus {
                ok: true,
                error: String::new(),
            }),
            vectors_count: points_count,
            indexed_vectors_count: points_count,
            points_count,
            segments_count: 1,
            config: Some(CollectionConfig {
                vectors: Some(VectorsConfig {
                    config: Some(vectors_config::Config::Params(VectorParams {
                        size: vector_dim,
                        distance: distance as i32,
                        on_disk: Some(false),
                    })),
                }),
                shard_number: 1,
                replication_factor: 1,
                hnsw_config: Some(HnswConfig {
                    m: 16,
                    ef_construct: 100,
                    full_scan_threshold: 10000,
                    max_indexing_threads: None,
                    on_disk: Some(false),
                }),
                wal_config: Some(WalConfig {
                    wal_capacity_mb: 32,
                    wal_segments_ahead: 0,
                }),
                optimizer_config: Some(OptimizerConfig {
                    deleted_threshold: 0.2,
                    vacuum_min_vector_number: 1000,
                    default_segment_number: 0,
                    max_segment_size: None,
                    memmap_threshold: None,
                    indexing_threshold: 20000,
                    flush_interval_sec: 5,
                }),
            }),
            payload_schema: Default::default(),
        };

        Ok(Response::new(GetCollectionInfoResponse {
            result: Some(result),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn list(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let start_time = Instant::now();
        let collections = self.storage.list_collections();
        
        let descriptions: Vec<CollectionDescription> = collections
            .into_iter()
            .map(|name| CollectionDescription { name })
            .collect();

        Ok(Response::new(ListCollectionsResponse {
            collections: descriptions,
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn create(
        &self,
        request: Request<CreateCollection>,
    ) -> Result<Response<CollectionOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let (vector_dim, distance) = if let Some(vectors_config) = req.vectors_config {
            match vectors_config.config {
                Some(vectors_config::Config::Params(params)) => {
                    let dist = match Distance::try_from(params.distance) {
                        Ok(Distance::Cosine) => CoreDistance::Cosine,
                        Ok(Distance::Euclid) => CoreDistance::Euclidean,
                        Ok(Distance::Dot) => CoreDistance::Dot,
                        _ => CoreDistance::Cosine,
                    };
                    (params.size as usize, dist)
                }
                Some(vectors_config::Config::ParamsMap(map)) => {
                    // Take first vector config
                    if let Some((_, params)) = map.map.into_iter().next() {
                        let dist = match Distance::try_from(params.distance) {
                            Ok(Distance::Cosine) => CoreDistance::Cosine,
                            Ok(Distance::Euclid) => CoreDistance::Euclidean,
                            Ok(Distance::Dot) => CoreDistance::Dot,
                            _ => CoreDistance::Cosine,
                        };
                        (params.size as usize, dist)
                    } else {
                        return Err(Status::invalid_argument("Vector configuration required"));
                    }
                }
                None => return Err(Status::invalid_argument("Vector configuration required")),
            }
        } else {
            return Err(Status::invalid_argument("Vector configuration required"));
        };

        let config = vectx_core::CollectionConfig {
            name: req.collection_name,
            vector_dim,
            distance,
            use_hnsw: true,
            enable_bm25: false,
        };

        self.storage.create_collection(config)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(CollectionOperationResponse {
            result: true,
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn update(
        &self,
        request: Request<UpdateCollection>,
    ) -> Result<Response<CollectionOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        // Verify collection exists
        if self.storage.get_collection(&req.collection_name).is_none() {
            return Err(Status::not_found("Collection not found"));
        }

        // Update not fully implemented - just return success
        Ok(Response::new(CollectionOperationResponse {
            result: true,
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn delete(
        &self,
        request: Request<DeleteCollection>,
    ) -> Result<Response<CollectionOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        match self.storage.delete_collection(&req.collection_name) {
            Ok(true) => Ok(Response::new(CollectionOperationResponse {
                result: true,
                time: start_time.elapsed().as_secs_f64(),
            })),
            Ok(false) => Err(Status::not_found("Collection not found")),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn collection_exists(
        &self,
        request: Request<CollectionExistsRequest>,
    ) -> Result<Response<CollectionExistsResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let exists = self.storage.collection_exists(&req.collection_name);

        Ok(Response::new(CollectionExistsResponse {
            result: Some(CollectionExistsResult { exists }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }
}

// ============================================================================
// Points Service
// ============================================================================

pub struct PointsService {
    storage: Arc<StorageManager>,
}

impl PointsService {
    pub fn new(storage: Arc<StorageManager>) -> Self {
        Self { storage }
    }

    fn parse_point_id(id: &vectx::PointId) -> Option<String> {
        match &id.point_id_options {
            Some(point_id::PointIdOptions::Num(n)) => Some(n.to_string()),
            Some(point_id::PointIdOptions::Uuid(s)) => Some(s.clone()),
            None => None,
        }
    }

    fn to_proto_point_id(id: &PointId) -> vectx::PointId {
        match id {
            PointId::String(s) => vectx::PointId {
                point_id_options: Some(point_id::PointIdOptions::Uuid(s.clone())),
            },
            PointId::Integer(i) => vectx::PointId {
                point_id_options: Some(point_id::PointIdOptions::Num(*i)),
            },
            PointId::Uuid(u) => vectx::PointId {
                point_id_options: Some(point_id::PointIdOptions::Uuid(u.to_string())),
            },
        }
    }

    fn proto_value_to_json(value: &vectx::Value) -> serde_json::Value {
        match &value.kind {
            Some(value::Kind::DoubleValue(v)) => serde_json::json!(*v),
            Some(value::Kind::IntegerValue(v)) => serde_json::json!(*v),
            Some(value::Kind::StringValue(v)) => serde_json::json!(v),
            Some(value::Kind::BoolValue(v)) => serde_json::json!(*v),
            Some(value::Kind::ListValue(list)) => {
                let values: Vec<serde_json::Value> = list.values.iter()
                    .map(Self::proto_value_to_json)
                    .collect();
                serde_json::json!(values)
            }
            Some(value::Kind::StructValue(s)) => {
                let map: serde_json::Map<String, serde_json::Value> = s.fields.iter()
                    .map(|(k, v)| (k.clone(), Self::proto_value_to_json(v)))
                    .collect();
                serde_json::Value::Object(map)
            }
            Some(value::Kind::NullValue(_)) | None => serde_json::Value::Null,
        }
    }

    fn json_to_proto_value(value: &serde_json::Value) -> vectx::Value {
        let kind = match value {
            serde_json::Value::Null => Some(value::Kind::NullValue(0)),
            serde_json::Value::Bool(b) => Some(value::Kind::BoolValue(*b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(value::Kind::IntegerValue(i))
                } else {
                    n.as_f64().map(value::Kind::DoubleValue)
                }
            }
            serde_json::Value::String(s) => Some(value::Kind::StringValue(s.clone())),
            serde_json::Value::Array(arr) => {
                let values = arr.iter().map(Self::json_to_proto_value).collect();
                Some(value::Kind::ListValue(ListValue { values }))
            }
            serde_json::Value::Object(map) => {
                let fields = map.iter()
                    .map(|(k, v)| (k.clone(), Self::json_to_proto_value(v)))
                    .collect();
                Some(value::Kind::StructValue(Struct { fields }))
            }
        };
        vectx::Value { kind }
    }
}

#[tonic::async_trait]
impl vectx::points_server::Points for PointsService {
    async fn upsert(
        &self,
        request: Request<UpsertPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        let points: Result<Vec<Point>, Status> = req.points.into_iter().map(|p| {
            let id = p.id.as_ref()
                .and_then(Self::parse_point_id)
                .ok_or_else(|| Status::invalid_argument("Point ID required"))?;
            
            let point_id = if let Ok(num) = id.parse::<u64>() {
                PointId::Integer(num)
            } else {
                PointId::String(id)
            };
            
            let vector_data = p.vectors.as_ref()
                .and_then(|vi| match &vi.variant {
                    Some(vector_input::Variant::Dense(v)) => Some(v.data.clone()),
                    Some(vector_input::Variant::Named(nv)) => {
                        nv.vectors.values().next().map(|v| v.data.clone())
                    }
                    None => None,
                })
                .ok_or_else(|| Status::invalid_argument("Vector required"))?;
            
            let payload = if p.payload.is_empty() {
                None
            } else {
                let json_map: serde_json::Map<String, serde_json::Value> = p.payload.iter()
                    .map(|(k, v)| (k.clone(), Self::proto_value_to_json(v)))
                    .collect();
                Some(serde_json::Value::Object(json_map))
            };
            
            let vector = Vector::new(vector_data);
            Ok(Point::new(point_id, vector, payload))
        }).collect();

        let points = points?;
        let count = points.len();
        
        if count > 1 {
            collection.batch_upsert(points)
                .map_err(|e| Status::internal(e.to_string()))?;
        } else if let Some(point) = points.into_iter().next() {
            collection.upsert(point)
                .map_err(|e| Status::internal(e.to_string()))?;
        }

        Ok(Response::new(PointsOperationResponse {
            result: Some(UpdateResult {
                operation_id: 0,
                status: UpdateStatus::Acknowledged as i32,
            }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn delete(
        &self,
        request: Request<DeletePoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        if let Some(points_selector) = req.points {
            if let Some(points_selector::PointsSelectorOneOf::Points(list)) = points_selector.points_selector_one_of {
                for point_id in list.ids {
                    if let Some(id_str) = Self::parse_point_id(&point_id) {
                        let _ = collection.delete(&id_str);
                    }
                }
            }
        }

        Ok(Response::new(PointsOperationResponse {
            result: Some(UpdateResult {
                operation_id: 0,
                status: UpdateStatus::Acknowledged as i32,
            }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn get(
        &self,
        request: Request<GetPoints>,
    ) -> Result<Response<GetResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        let mut results = Vec::new();
        for point_id in req.ids {
            if let Some(id_str) = Self::parse_point_id(&point_id) {
                if let Some(point) = collection.get(&id_str) {
                    let payload: std::collections::HashMap<String, vectx::Value> = point.payload
                        .as_ref()
                        .and_then(|p| p.as_object())
                        .map(|obj| {
                            obj.iter()
                                .map(|(k, v)| (k.clone(), Self::json_to_proto_value(v)))
                                .collect()
                        })
                        .unwrap_or_default();

                    results.push(RetrievedPoint {
                        id: Some(Self::to_proto_point_id(&point.id)),
                        payload,
                        vectors: Some(VectorInput {
                            variant: Some(vector_input::Variant::Dense(vectx::Vector {
                                data: point.vector.as_slice().to_vec(),
                            })),
                        }),
                    });
                }
            }
        }

        Ok(Response::new(GetResponse {
            result: results,
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn set_payload(
        &self,
        request: Request<SetPayloadPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        if self.storage.get_collection(&req.collection_name).is_none() {
            return Err(Status::not_found("Collection not found"));
        }

        // Payload update stub - not fully implemented
        Ok(Response::new(PointsOperationResponse {
            result: Some(UpdateResult {
                operation_id: 0,
                status: UpdateStatus::Acknowledged as i32,
            }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn delete_payload(
        &self,
        request: Request<DeletePayloadPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        if self.storage.get_collection(&req.collection_name).is_none() {
            return Err(Status::not_found("Collection not found"));
        }

        Ok(Response::new(PointsOperationResponse {
            result: Some(UpdateResult {
                operation_id: 0,
                status: UpdateStatus::Acknowledged as i32,
            }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn clear_payload(
        &self,
        request: Request<ClearPayloadPoints>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        if self.storage.get_collection(&req.collection_name).is_none() {
            return Err(Status::not_found("Collection not found"));
        }

        Ok(Response::new(PointsOperationResponse {
            result: Some(UpdateResult {
                operation_id: 0,
                status: UpdateStatus::Acknowledged as i32,
            }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn create_field_index(
        &self,
        request: Request<CreateFieldIndexCollection>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        if self.storage.get_collection(&req.collection_name).is_none() {
            return Err(Status::not_found("Collection not found"));
        }

        Ok(Response::new(PointsOperationResponse {
            result: Some(UpdateResult {
                operation_id: 0,
                status: UpdateStatus::Acknowledged as i32,
            }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn delete_field_index(
        &self,
        request: Request<DeleteFieldIndexCollection>,
    ) -> Result<Response<PointsOperationResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        if self.storage.get_collection(&req.collection_name).is_none() {
            return Err(Status::not_found("Collection not found"));
        }

        Ok(Response::new(PointsOperationResponse {
            result: Some(UpdateResult {
                operation_id: 0,
                status: UpdateStatus::Acknowledged as i32,
            }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn search(
        &self,
        request: Request<SearchPoints>,
    ) -> Result<Response<SearchResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        let query = Vector::new(req.vector);
        let limit = req.limit as usize;
        
        let results = collection.search(&query, limit, None);
        
        let scored_points: Vec<ScoredPoint> = results.into_iter().map(|(point, score)| {
            let payload: std::collections::HashMap<String, vectx::Value> = point.payload
                .as_ref()
                .and_then(|p| p.as_object())
                .map(|obj| {
                    obj.iter()
                        .map(|(k, v)| (k.clone(), Self::json_to_proto_value(v)))
                        .collect()
                })
                .unwrap_or_default();

            ScoredPoint {
                id: Some(Self::to_proto_point_id(&point.id)),
                payload,
                score,
                vectors: None,
                version: Some(0),
            }
        }).collect();

        Ok(Response::new(SearchResponse {
            result: scored_points,
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn scroll(
        &self,
        request: Request<ScrollPoints>,
    ) -> Result<Response<ScrollResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        let limit = req.limit.unwrap_or(10) as usize;
        let all_points = collection.get_all_points();
        
        // Get offset
        let offset_id: Option<String> = req.offset.as_ref()
            .and_then(Self::parse_point_id);
        
        let mut points_iter = all_points.iter();
        
        // Skip to offset
        if let Some(ref offset) = offset_id {
            for p in points_iter.by_ref() {
                if p.id.to_string() == *offset {
                    break;
                }
            }
        }
        
        let mut results = Vec::new();
        let mut last_id = None;
        
        for point in points_iter.take(limit) {
            last_id = Some(Self::to_proto_point_id(&point.id));
            
            let payload: std::collections::HashMap<String, vectx::Value> = point.payload
                .as_ref()
                .and_then(|p| p.as_object())
                .map(|obj| {
                    obj.iter()
                        .map(|(k, v)| (k.clone(), Self::json_to_proto_value(v)))
                        .collect()
                })
                .unwrap_or_default();

            results.push(RetrievedPoint {
                id: Some(Self::to_proto_point_id(&point.id)),
                payload,
                vectors: Some(VectorInput {
                    variant: Some(vector_input::Variant::Dense(vectx::Vector {
                        data: point.vector.as_slice().to_vec(),
                    })),
                }),
            });
        }

        // Determine next page offset
        let next_offset = if results.len() == limit {
            last_id
        } else {
            None
        };

        Ok(Response::new(ScrollResponse {
            next_page_offset: next_offset,
            result: results,
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn recommend(
        &self,
        request: Request<RecommendPoints>,
    ) -> Result<Response<RecommendResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        let limit = req.limit as usize;
        
        // Collect positive vectors
        let mut positive_vectors: Vec<Vec<f32>> = Vec::new();
        let mut exclude_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        
        for pos_id in &req.positive {
            if let Some(id_str) = Self::parse_point_id(pos_id) {
                exclude_ids.insert(id_str.clone());
                if let Some(point) = collection.get(&id_str) {
                    positive_vectors.push(point.vector.as_slice().to_vec());
                }
            }
        }
        
        if positive_vectors.is_empty() {
            return Err(Status::invalid_argument("At least one valid positive example required"));
        }
        
        // Collect negative vectors
        let mut negative_vectors: Vec<Vec<f32>> = Vec::new();
        for neg_id in &req.negative {
            if let Some(id_str) = Self::parse_point_id(neg_id) {
                exclude_ids.insert(id_str.clone());
                if let Some(point) = collection.get(&id_str) {
                    negative_vectors.push(point.vector.as_slice().to_vec());
                }
            }
        }
        
        // Compute average positive
        let dim = positive_vectors[0].len();
        let mut avg_positive = vec![0.0f32; dim];
        for vec in &positive_vectors {
            for (i, &val) in vec.iter().enumerate() {
                if i < dim { avg_positive[i] += val; }
            }
        }
        let pos_count = positive_vectors.len() as f32;
        for val in &mut avg_positive { *val /= pos_count; }
        
        // Create query vector
        let query_data = if !negative_vectors.is_empty() {
            let mut avg_negative = vec![0.0f32; dim];
            for vec in &negative_vectors {
                for (i, &val) in vec.iter().enumerate() {
                    if i < dim { avg_negative[i] += val; }
                }
            }
            let neg_count = negative_vectors.len() as f32;
            for val in &mut avg_negative { *val /= neg_count; }
            
            avg_positive.iter()
                .zip(avg_negative.iter())
                .map(|(p, n)| 2.0 * p - n)
                .collect()
        } else {
            avg_positive
        };
        
        let query = Vector::new(query_data);
        let search_results = collection.search(&query, limit + exclude_ids.len(), None);
        
        let scored_points: Vec<ScoredPoint> = search_results
            .into_iter()
            .filter(|(point, _)| !exclude_ids.contains(&point.id.to_string()))
            .take(limit)
            .map(|(point, score)| {
                let payload: std::collections::HashMap<String, vectx::Value> = point.payload
                    .as_ref()
                    .and_then(|p| p.as_object())
                    .map(|obj| {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), Self::json_to_proto_value(v)))
                            .collect()
                    })
                    .unwrap_or_default();

                ScoredPoint {
                    id: Some(Self::to_proto_point_id(&point.id)),
                    payload,
                    score,
                    vectors: None,
                    version: Some(0),
                }
            })
            .collect();

        Ok(Response::new(RecommendResponse {
            result: scored_points,
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn count(
        &self,
        request: Request<CountPoints>,
    ) -> Result<Response<CountResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        Ok(Response::new(CountResponse {
            result: Some(CountResult {
                count: collection.count() as u64,
            }),
            time: start_time.elapsed().as_secs_f64(),
        }))
    }

    async fn query(
        &self,
        request: Request<QueryPoints>,
    ) -> Result<Response<QueryResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        let collection = self.storage.get_collection(&req.collection_name)
            .ok_or_else(|| Status::not_found("Collection not found"))?;

        let limit = req.limit as usize;
        
        let query_data = req.query
            .and_then(|vi| match vi.variant {
                Some(vector_input::Variant::Dense(v)) => Some(v.data),
                Some(vector_input::Variant::Named(nv)) => {
                    nv.vectors.values().next().map(|v| v.data.clone())
                }
                None => None,
            })
            .ok_or_else(|| Status::invalid_argument("Query vector required"))?;
        
        let query = Vector::new(query_data);
        let results = collection.search(&query, limit, None);
        
        let scored_points: Vec<ScoredPoint> = results.into_iter().map(|(point, score)| {
            let payload: std::collections::HashMap<String, vectx::Value> = point.payload
                .as_ref()
                .and_then(|p| p.as_object())
                .map(|obj| {
                    obj.iter()
                        .map(|(k, v)| (k.clone(), Self::json_to_proto_value(v)))
                        .collect()
                })
                .unwrap_or_default();

            ScoredPoint {
                id: Some(Self::to_proto_point_id(&point.id)),
                payload,
                score,
                vectors: None,
                version: Some(0),
            }
        }).collect();

        Ok(Response::new(QueryResponse {
            result: scored_points,
            time: start_time.elapsed().as_secs_f64(),
        }))
    }
}

// ============================================================================
// Snapshots Service
// ============================================================================

pub struct SnapshotsService {
    storage: Arc<StorageManager>,
}

impl SnapshotsService {
    pub fn new(storage: Arc<StorageManager>) -> Self {
        Self { storage }
    }
}

#[tonic::async_trait]
impl vectx::snapshots_server::Snapshots for SnapshotsService {
    async fn create(
        &self,
        request: Request<CreateSnapshotRequest>,
    ) -> Result<Response<CreateSnapshotResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        match self.storage.create_collection_snapshot(&req.collection_name) {
            Ok(snapshot) => Ok(Response::new(CreateSnapshotResponse {
                result: Some(SnapshotDescription {
                    name: snapshot.name,
                    creation_time: snapshot.creation_time.unwrap_or_default(),
                    size: snapshot.size as i64,
                    checksum: snapshot.checksum,
                }),
                time: start_time.elapsed().as_secs_f64(),
            })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn list(
        &self,
        request: Request<ListSnapshotsRequest>,
    ) -> Result<Response<ListSnapshotsResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        match self.storage.list_collection_snapshots(&req.collection_name) {
            Ok(snapshots) => {
                let descriptions: Vec<SnapshotDescription> = snapshots
                    .into_iter()
                    .map(|s| SnapshotDescription {
                        name: s.name,
                        creation_time: s.creation_time.unwrap_or_default(),
                        size: s.size as i64,
                        checksum: s.checksum,
                    })
                    .collect();
                
                Ok(Response::new(ListSnapshotsResponse {
                    snapshots: descriptions,
                    time: start_time.elapsed().as_secs_f64(),
                }))
            }
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn delete(
        &self,
        request: Request<DeleteSnapshotRequest>,
    ) -> Result<Response<DeleteSnapshotResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        match self.storage.delete_collection_snapshot(&req.collection_name, &req.snapshot_name) {
            Ok(true) => Ok(Response::new(DeleteSnapshotResponse {
                result: true,
                time: start_time.elapsed().as_secs_f64(),
            })),
            Ok(false) => Err(Status::not_found("Snapshot not found")),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }

    async fn recover(
        &self,
        request: Request<RecoverSnapshotRequest>,
    ) -> Result<Response<RecoverSnapshotResponse>, Status> {
        let start_time = Instant::now();
        let req = request.into_inner();
        
        match self.storage.recover_from_snapshot(&req.collection_name, &req.location) {
            Ok(_) => Ok(Response::new(RecoverSnapshotResponse {
                result: true,
                time: start_time.elapsed().as_secs_f64(),
            })),
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }
}

// ============================================================================
// gRPC Server Startup
// ============================================================================

pub struct GrpcApi;

impl GrpcApi {
    pub async fn start(
        storage: Arc<StorageManager>,
        port: u16,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let addr = format!("0.0.0.0:{}", port).parse()?;
        
        let qdrant_service = vectx::qdrant_server::QdrantServer::new(QdrantService);
        let collections_service = vectx::collections_server::CollectionsServer::new(
            CollectionsService::new(storage.clone())
        );
        let points_service = vectx::points_server::PointsServer::new(
            PointsService::new(storage.clone())
        );
        let snapshots_service = vectx::snapshots_server::SnapshotsServer::new(
            SnapshotsService::new(storage)
        );
        
        println!("gRPC server listening on {}", addr);
        
        tonic::transport::Server::builder()
            .add_service(qdrant_service)
            .add_service(collections_service)
            .add_service(points_service)
            .add_service(snapshots_service)
            .serve(addr)
            .await?;
        
        Ok(())
    }
}
