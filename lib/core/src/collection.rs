use crate::{Error, Point, Result, Vector, HnswIndex, BM25Index, Filter, MultiVector};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Configuration for a collection
#[derive(Debug, Clone)]
pub struct CollectionConfig {
    pub name: String,
    pub vector_dim: usize,
    pub distance: Distance,
    pub use_hnsw: bool,
    pub enable_bm25: bool,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            vector_dim: 128,
            distance: Distance::Cosine,
            use_hnsw: true,
            enable_bm25: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Cosine,
    Euclidean,
    Dot,
}

/// Payload field index type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadIndexType {
    Keyword,
    Integer,
    Float,
    Bool,
    Geo,
    Text,
}

/// A collection of vectors with metadata
pub struct Collection {
    config: CollectionConfig,
    points: Arc<RwLock<HashMap<String, Point>>>,
    hnsw: Option<Arc<RwLock<HnswIndex>>>,
    bm25: Option<Arc<RwLock<BM25Index>>>,
    hnsw_built: Arc<RwLock<bool>>,
    hnsw_rebuilding: Arc<AtomicBool>,
    batch_mode: Arc<RwLock<bool>>,
    pending_points: Arc<RwLock<Vec<Point>>>,
    /// Payload field indexes
    payload_indexes: Arc<RwLock<HashMap<String, PayloadIndexType>>>,
    /// Operation counter for tracking write operations
    operation_counter: Arc<std::sync::atomic::AtomicU64>,
}

impl Collection {
    pub fn new(config: CollectionConfig) -> Self {
        let hnsw = if config.use_hnsw {
            Some(Arc::new(RwLock::new(HnswIndex::new(16, 3))))
        } else {
            None
        };

        let bm25 = if config.enable_bm25 {
            Some(Arc::new(RwLock::new(BM25Index::new())))
        } else {
            None
        };

        Self {
            config,
            points: Arc::new(RwLock::new(HashMap::new())),
            hnsw,
            bm25,
            hnsw_built: Arc::new(RwLock::new(false)),
            hnsw_rebuilding: Arc::new(AtomicBool::new(false)),
            batch_mode: Arc::new(RwLock::new(false)),
            pending_points: Arc::new(RwLock::new(Vec::new())),
            payload_indexes: Arc::new(RwLock::new(HashMap::new())),
            operation_counter: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
    
    /// Get next operation ID (atomically increments counter)
    #[inline]
    pub fn next_operation_id(&self) -> u64 {
        self.operation_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.config.name
    }

    #[inline]
    #[must_use]
    pub fn vector_dim(&self) -> usize {
        self.config.vector_dim
    }

    #[inline]
    #[must_use]
    pub fn distance(&self) -> Distance {
        self.config.distance
    }

    #[inline]
    #[must_use]
    pub fn use_hnsw(&self) -> bool {
        self.config.use_hnsw
    }

    #[inline]
    #[must_use]
    pub fn enable_bm25(&self) -> bool {
        self.config.enable_bm25
    }

    #[inline]
    #[must_use]
    pub fn count(&self) -> usize {
        self.points.read().len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.read().is_empty()
    }

    /// Get all points in the collection
    pub fn get_all_points(&self) -> Vec<Point> {
        self.points.read().values().cloned().collect()
    }

    /// Insert or update a point
    pub fn upsert(&self, point: Point) -> Result<()> {
        // Skip dimension check for sparse-only collections (vector_dim == 0)
        if self.config.vector_dim > 0 && point.vector.dim() != self.config.vector_dim {
            return Err(Error::InvalidDimension {
                expected: self.config.vector_dim,
                actual: point.vector.dim(),
            });
        }

        let id_str = point.id.to_string();
        
        // Check if point exists and get its current version
        let new_version = {
            let points = self.points.read();
            if let Some(existing) = points.get(&id_str) {
                existing.version + 1
            } else {
                0
            }
        };
        
        // Create point with updated version
        let mut versioned_point = point;
        versioned_point.version = new_version;
        
        let in_batch = *self.batch_mode.read();
        if in_batch {
            self.points.write().insert(id_str.clone(), versioned_point.clone());
            self.pending_points.write().push(versioned_point);
            return Ok(());
        }
        
        if let Some(hnsw) = &self.hnsw {
            let built = *self.hnsw_built.read();
            if built {
                let mut normalized_point = versioned_point.clone();
                normalized_point.vector.normalize();
                
                let mut index = hnsw.write();
                index.insert(normalized_point);
            }
        }

        if let Some(bm25) = &self.bm25 {
            if let Some(payload) = &versioned_point.payload {
                if let Some(text) = payload.get("text").and_then(|v| v.as_str()) {
                    let mut index = bm25.write();
                    index.insert_doc(&id_str, text);
                }
            }
        }

        self.points.write().insert(id_str, versioned_point);
        Ok(())
    }

    /// Start batch insert mode
    pub fn start_batch(&self) {
        *self.batch_mode.write() = true;
        self.pending_points.write().clear();
    }

    /// End batch insert mode
    pub fn end_batch(&self) -> Result<()> {
        *self.batch_mode.write() = false;
        
        if let Some(hnsw) = &self.hnsw {
            let points = self.points.read();
            let point_count = points.len();
            
            const HNSW_REBUILD_THRESHOLD: usize = 10_000;
            
            if point_count > HNSW_REBUILD_THRESHOLD && !self.hnsw_rebuilding.load(Ordering::Acquire) {
                self.hnsw_rebuilding.store(true, Ordering::Release);
                let points_clone: Vec<Point> = points.values().cloned().collect();
                let hnsw_clone = hnsw.clone();
                let built_flag = self.hnsw_built.clone();
                let rebuilding_flag = self.hnsw_rebuilding.clone();
                
                let job = crate::background::HnswRebuildJob::new(
                    points_clone,
                    hnsw_clone,
                    built_flag,
                    rebuilding_flag,
                );
                crate::background::get_background_system().submit(Box::new(job));
            }
        }
        
        self.pending_points.write().clear();
        Ok(())
    }

    /// Batch insert multiple points
    pub fn batch_upsert(&self, points: Vec<Point>) -> Result<()> {
        self.start_batch();
        for point in points {
            self.upsert(point)?;
        }
        self.end_batch()?;
        Ok(())
    }

    /// Batch insert with optional pre-warming
    pub fn batch_upsert_with_prewarm(&self, points: Vec<Point>, prewarm: bool) -> Result<()> {
        self.batch_upsert(points)?;
        if prewarm {
            self.prewarm_index()?;
        }
        Ok(())
    }

    /// Get a point by ID
    #[inline]
    pub fn get(&self, id: &str) -> Option<Point> {
        self.points.read().get(id).cloned()
    }

    /// Delete a point by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        if let Some(hnsw) = &self.hnsw {
            let mut index = hnsw.write();
            index.remove(id);
        }

        if let Some(bm25) = &self.bm25 {
            let mut index = bm25.write();
            index.delete_doc(id);
        }

        let mut points = self.points.write();
        Ok(points.remove(id).is_some())
    }

    /// Set payload values for a point (merge with existing)
    pub fn set_payload(&self, id: &str, payload: serde_json::Value) -> Result<bool> {
        let mut points = self.points.write();
        if let Some(point) = points.get_mut(id) {
            if let Some(existing) = &mut point.payload {
                if let (Some(existing_obj), Some(new_obj)) = (existing.as_object_mut(), payload.as_object()) {
                    for (key, value) in new_obj {
                        existing_obj.insert(key.clone(), value.clone());
                    }
                }
            } else {
                point.payload = Some(payload);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Overwrite entire payload for a point
    pub fn overwrite_payload(&self, id: &str, payload: serde_json::Value) -> Result<bool> {
        let mut points = self.points.write();
        if let Some(point) = points.get_mut(id) {
            point.payload = Some(payload);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete specific payload keys from a point
    pub fn delete_payload_keys(&self, id: &str, keys: &[String]) -> Result<bool> {
        let mut points = self.points.write();
        if let Some(point) = points.get_mut(id) {
            if let Some(payload) = &mut point.payload {
                if let Some(obj) = payload.as_object_mut() {
                    for key in keys {
                        obj.remove(key);
                    }
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clear all payload from a point
    pub fn clear_payload(&self, id: &str) -> Result<bool> {
        let mut points = self.points.write();
        if let Some(point) = points.get_mut(id) {
            point.payload = None;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Update vector for a point
    pub fn update_vector(&self, id: &str, vector: Vector) -> Result<bool> {
        let mut points = self.points.write();
        if let Some(point) = points.get_mut(id) {
            point.vector = vector.clone();
            
            // Update HNSW index if present
            if let Some(hnsw) = &self.hnsw {
                let mut index = hnsw.write();
                index.remove(id);
                // Insert the updated point
                index.insert(point.clone());
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Update multivector for a point
    pub fn update_multivector(&self, id: &str, multivector: Option<MultiVector>) -> Result<bool> {
        let mut points = self.points.write();
        if let Some(point) = points.get_mut(id) {
            point.multivector = multivector;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete vector (set to empty) - for named vectors this would delete specific vector
    pub fn delete_vector(&self, id: &str) -> Result<bool> {
        // For now, deleting a vector means deleting the point
        // In full implementation, named vectors would be individually deletable
        self.delete(id)
    }

    /// Create a payload field index
    pub fn create_payload_index(&self, field_name: &str, index_type: PayloadIndexType) -> Result<bool> {
        let mut indexes = self.payload_indexes.write();
        indexes.insert(field_name.to_string(), index_type);
        Ok(true)
    }

    /// Delete a payload field index
    pub fn delete_payload_index(&self, field_name: &str) -> Result<bool> {
        let mut indexes = self.payload_indexes.write();
        Ok(indexes.remove(field_name).is_some())
    }

    /// Get all payload indexes
    pub fn get_payload_indexes(&self) -> HashMap<String, PayloadIndexType> {
        self.payload_indexes.read().clone()
    }

    /// Check if a field is indexed
    pub fn is_field_indexed(&self, field_name: &str) -> bool {
        self.payload_indexes.read().contains_key(field_name)
    }

    /// Pre-warm HNSW index
    pub fn prewarm_index(&self) -> Result<()> {
        if let Some(hnsw) = &self.hnsw {
            let mut built = self.hnsw_built.write();
            if !*built {
                let points = self.points.read();
                if !points.is_empty() {
                    let mut index = hnsw.write();
                    *index = HnswIndex::new(16, 3);
                    for point in points.values() {
                        index.insert(point.clone());
                    }
                    *built = true;
                }
            }
        }
        Ok(())
    }

    /// Fast brute-force search using SIMD - optimal for small datasets
    fn brute_force_search(&self, query: &Vector, limit: usize, filter: Option<&dyn Filter>) -> Vec<(Point, f32)> {
        use rayon::prelude::*;
        
        let points = self.points.read();
        let query_slice = query.as_slice();
        let distance = self.config.distance;
        
        // Collect points to a Vec for indexing
        let point_vec: Vec<_> = points.values().collect();
        
        // Parallel scoring - compute scores without cloning points
        // Only clone the final top-k results
        // Use parallel only for 10K+ vectors (rayon has overhead)
        let scored: Vec<(usize, f32)> = if point_vec.len() >= 10000 && filter.is_none() {
            // Parallel path for larger datasets without filter
            point_vec
                .par_iter()
                .enumerate()
                .map(|(idx, point)| {
                    let score = match distance {
                        Distance::Cosine => {
                            crate::simd::dot_product_simd(query_slice, point.vector.as_slice())
                        }
                        Distance::Euclidean => {
                            -crate::simd::l2_distance_simd(query_slice, point.vector.as_slice())
                        }
                        Distance::Dot => {
                            crate::simd::dot_product_simd(query_slice, point.vector.as_slice())
                        }
                    };
                    (idx, score)
                })
                .collect()
        } else {
            // Sequential path - optimized for common case (Cosine without filter)
            let mut results = Vec::with_capacity(point_vec.len());
            
            if filter.is_none() && matches!(distance, Distance::Cosine) {
                // Hot path: Cosine without filter - avoid branching
                for (idx, point) in point_vec.iter().enumerate() {
                    let score = crate::simd::dot_product_simd(query_slice, point.vector.as_slice());
                    results.push((idx, score));
                }
            } else {
                // General path with filter/distance checks
                for (idx, point) in point_vec.iter().enumerate() {
                    if let Some(f) = filter {
                        if !f.matches(point) {
                            continue;
                        }
                    }
                    
                    let score = match distance {
                        Distance::Cosine => {
                            crate::simd::dot_product_simd(query_slice, point.vector.as_slice())
                        }
                        Distance::Euclidean => {
                            -crate::simd::l2_distance_simd(query_slice, point.vector.as_slice())
                        }
                        Distance::Dot => {
                            crate::simd::dot_product_simd(query_slice, point.vector.as_slice())
                        }
                    };
                    
                    results.push((idx, score));
                }
            }
            results
        };
        
        // Get top-k using partial sort
        let mut scored = scored;
        if scored.len() > limit {
            scored.select_nth_unstable_by(limit, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(limit);
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Only clone the top-k points (avoiding cloning all points)
        scored
            .into_iter()
            .map(|(idx, score)| (point_vec[idx].clone(), score))
            .collect()
    }

    /// Search for similar vectors
    /// Uses brute-force for small datasets (<1000), HNSW for larger ones
    pub fn search(
        &self,
        query: &Vector,
        limit: usize,
        filter: Option<&dyn Filter>,
    ) -> Vec<(Point, f32)> {
        let normalized_query = query.normalized();
        let point_count = self.points.read().len();
        
        // Use brute-force for datasets up to 10K - SIMD is very fast and avoids HNSW overhead
        const BRUTE_FORCE_THRESHOLD: usize = 10000;
        if point_count < BRUTE_FORCE_THRESHOLD {
            return self.brute_force_search(&normalized_query, limit, filter);
        }
        
        if let Some(hnsw) = &self.hnsw {
            // Check if we need to build the index first
            {
                let mut built = self.hnsw_built.write();
                if !*built {
                    let points = self.points.read();
                    if !points.is_empty() {
                        let mut index = hnsw.write();
                        *index = HnswIndex::new(16, 3);
                        for point in points.values() {
                            index.insert(point.clone());
                        }
                        *built = true;
                    }
                }
            }
            
            // Use write lock for search (HNSW search is now mutable for performance)
            let mut index = hnsw.write();
            let mut results = index.search(&normalized_query, limit, None);
            
            if let Some(f) = filter {
                results.retain(|(point, _)| f.matches(point));
            }
            
            results
        } else {
            let points = self.points.read();
            let results: Vec<(Point, f32)> = points
                .values()
                .filter(|point| {
                    filter.map(|f| f.matches(point)).unwrap_or(true)
                })
                .map(|point| {
                    let score = match self.config.distance {
                        Distance::Cosine => point.vector.cosine_similarity(query),
                        Distance::Euclidean => -point.vector.l2_distance(query),
                        Distance::Dot => {
                            point.vector.as_slice()
                                .iter()
                                .zip(query.as_slice().iter())
                                .map(|(a, b)| a * b)
                                .sum()
                        }
                    };
                    (point.clone(), score)
                })
                .collect();

            let mut sorted = results;
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sorted.truncate(limit);
            sorted
        }
    }

    /// BM25 text search
    pub fn search_text(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        if let Some(bm25) = &self.bm25 {
            let index = bm25.read();
            index.search(query, limit)
        } else {
            Vec::new()
        }
    }
    
    /// Search using multivector MaxSim scoring (ColBERT-style)
    /// 
    /// For each sub-vector in the query, finds the maximum similarity 
    /// with any sub-vector in each document, then sums all maximums.
    pub fn search_multivector(
        &self,
        query: &MultiVector,
        limit: usize,
        filter: Option<&dyn Filter>,
    ) -> Vec<(Point, f32)> {
        let points = self.points.read();
        
        let mut results: Vec<(Point, f32)> = Vec::with_capacity(points.len().min(limit * 2));
        
        for point in points.values() {
            if let Some(f) = filter {
                if !f.matches(point) {
                    continue;
                }
            }
            
            // Calculate MaxSim score
            let score = if let Some(doc_mv) = &point.multivector {
                // Both query and document have multivectors - use MaxSim
                match self.config.distance {
                    Distance::Cosine => query.max_sim_cosine(doc_mv),
                    Distance::Euclidean => query.max_sim_l2(doc_mv),
                    Distance::Dot => query.max_sim(doc_mv),
                }
            } else {
                // Document has single vector - wrap it as multivector
                let doc_mv = MultiVector::from_single(point.vector.as_slice().to_vec())
                    .unwrap_or_else(|_| MultiVector::new(vec![vec![0.0; query.dim()]]).unwrap());
                match self.config.distance {
                    Distance::Cosine => query.max_sim_cosine(&doc_mv),
                    Distance::Euclidean => query.max_sim_l2(&doc_mv),
                    Distance::Dot => query.max_sim(&doc_mv),
                }
            };
            
            results.push((point.clone(), score));
        }
        
        // Sort by score descending
        if results.len() > limit {
            results.select_nth_unstable_by(limit, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(limit);
        }
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get all points
    pub fn iter(&self) -> Vec<Point> {
        self.points.read().values().cloned().collect()
    }
    
    /// Search using sparse vectors (dot product on matching indices)
    pub fn search_sparse(
        &self,
        query: &crate::point::SparseVector,
        vector_name: &str,
        limit: usize,
        filter: Option<&dyn Filter>,
    ) -> Vec<(Point, f32)> {
        let points = self.points.read();
        
        let mut results: Vec<(Point, f32)> = Vec::with_capacity(points.len().min(limit * 2));
        
        for point in points.values() {
            // Apply filter if provided
            if let Some(f) = filter {
                if !f.matches(point) {
                    continue;
                }
            }
            
            // Get the named sparse vector from the point
            if let Some(point_sparse) = point.sparse_vectors.get(vector_name) {
                // Calculate dot product score
                let score = query.dot(point_sparse);
                
                // Only include if score > 0 (at least one matching index)
                if score > 0.0 {
                    results.push((point.clone(), score));
                }
            }
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        
        results
    }
}

