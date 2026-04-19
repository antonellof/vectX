use crate::{Point, Vector};
use std::collections::HashMap;
use std::cmp::Ordering;
use std::sync::Arc;
use parking_lot::RwLock;

/// Fast bit vector for visited node tracking
/// Much faster than HashSet for dense integer sets
#[derive(Clone)]
struct VisitedSet {
    bits: Vec<u64>,
    generation: u64,
    generations: Vec<u64>,
}

impl VisitedSet {
    #[inline]
    fn new(capacity: usize) -> Self {
        let num_words = capacity.div_ceil(64);
        Self {
            bits: vec![0; num_words],
            generation: 1,
            generations: vec![0; num_words],
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.generation += 1;
        // Only reset if we've wrapped around
        if self.generation == 0 {
            self.generation = 1;
            self.bits.fill(0);
            self.generations.fill(0);
        }
    }

    #[inline]
    fn ensure_capacity(&mut self, capacity: usize) {
        let num_words = capacity.div_ceil(64);
        if num_words > self.bits.len() {
            self.bits.resize(num_words, 0);
            self.generations.resize(num_words, 0);
        }
    }

    #[inline]
    fn insert(&mut self, idx: usize) -> bool {
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        let mask = 1u64 << bit_idx;

        if word_idx >= self.bits.len() {
            self.ensure_capacity(idx + 1);
        }

        // Check if this generation has been visited
        if self.generations[word_idx] != self.generation {
            self.bits[word_idx] = 0;
            self.generations[word_idx] = self.generation;
        }

        let was_set = (self.bits[word_idx] & mask) != 0;
        self.bits[word_idx] |= mask;
        !was_set
    }

    #[inline]
    #[allow(dead_code)] // Kept for parity with `set`; reserved for future debug/inspection tools
    fn contains(&self, idx: usize) -> bool {
        let word_idx = idx / 64;
        let bit_idx = idx % 64;
        
        if word_idx >= self.bits.len() {
            return false;
        }
        
        if self.generations[word_idx] != self.generation {
            return false;
        }
        
        (self.bits[word_idx] & (1u64 << bit_idx)) != 0
    }
}

/// Candidate for search with distance
#[derive(Clone, Copy)]
struct Candidate {
    idx: usize,
    dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.idx == other.idx
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reverse candidate for max-heap (furthest first)
#[derive(Clone, Copy)]
struct ReverseCandidate {
    idx: usize,
    dist: f32,
}

impl PartialEq for ReverseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.idx == other.idx
    }
}

impl Eq for ReverseCandidate {}

impl Ord for ReverseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ReverseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct HnswNode {
    point: Point,
    layers: Vec<Vec<usize>>,
}

/// High-performance HNSW index for approximate nearest neighbor search
/// Optimized with:
/// - Bit vector for O(1) visited tracking
/// - Contiguous vector storage for cache locality  
/// - Prefetching for reduced cache misses
/// - Optimized SIMD distance calculations
pub struct HnswIndex {
    nodes: Vec<HnswNode>,
    /// Contiguous storage for all vectors (cache-friendly)
    vectors: Vec<f32>,
    /// Dimension of vectors
    dim: usize,
    point_id_to_index: Arc<RwLock<HashMap<String, usize>>>,
    max_connections: usize,
    max_layers: usize,
    ef_construction: usize,
    /// Reusable visited set (avoid allocations)
    visited: VisitedSet,
}

impl HnswIndex {
    pub fn new(max_connections: usize, max_layers: usize) -> Self {
        Self {
            nodes: Vec::new(),
            vectors: Vec::new(),
            dim: 0,
            point_id_to_index: Arc::new(RwLock::new(HashMap::new())),
            max_connections,
            max_layers,
            ef_construction: 200,
            visited: VisitedSet::new(1024),
        }
    }

    /// Get vector slice for a node (from contiguous storage)
    #[inline(always)]
    fn get_vector(&self, node_idx: usize) -> &[f32] {
        let start = node_idx * self.dim;
        unsafe {
            // Safety: we maintain invariant that vectors has dim * nodes.len() elements
            self.vectors.get_unchecked(start..start + self.dim)
        }
    }

    /// Select layer using exponential decay
    #[inline]
    fn select_layer(&self) -> usize {
        let mut layer = 0;
        while layer < self.max_layers - 1 && rand::random::<f32>() < 0.5 {
            layer += 1;
        }
        layer
    }

    /// Optimized distance calculation using contiguous storage
    #[inline(always)]
    fn distance_to_node(&self, query: &[f32], node_idx: usize) -> f32 {
        let node_vec = self.get_vector(node_idx);
        let dot = crate::simd::dot_product_simd(query, node_vec);
        1.0 - dot
    }

    /// Prefetch vector data for a node (reduce cache misses)
    #[inline(always)]
    fn prefetch_node(&self, node_idx: usize) {
        if node_idx < self.nodes.len() {
            let start = node_idx * self.dim;
            if start < self.vectors.len() {
                #[cfg(target_arch = "x86_64")]
                {
                    let ptr = unsafe { self.vectors.as_ptr().add(start) };
                    unsafe {
                        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                    }
                }
                // On ARM, just touch the memory to bring it into cache
                #[cfg(target_arch = "aarch64")]
                {
                    let _ = unsafe { *self.vectors.as_ptr().add(start) };
                }
            }
        }
    }

    /// Find nearest neighbors using greedy search
    /// Optimized with bit vector, prefetching, and minimal allocations
    fn search_layer(
        &mut self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        use std::collections::BinaryHeap;

        // Clear and ensure capacity of visited set
        self.visited.clear();
        self.visited.ensure_capacity(self.nodes.len());

        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<ReverseCandidate> = BinaryHeap::with_capacity(ef + 1);

        let entry_dist = self.distance_to_node(query, entry_point);
        candidates.push(Candidate { idx: entry_point, dist: entry_dist });
        results.push(ReverseCandidate { idx: entry_point, dist: entry_dist });
        self.visited.insert(entry_point);

        // Cache worst distance for fast comparison
        let mut worst_dist = entry_dist;
        
        // Pre-allocated buffer for neighbor indices to avoid repeated allocations
        let mut neighbor_buffer: Vec<usize> = Vec::with_capacity(64);

        while let Some(Candidate { idx: current_idx, dist: current_dist }) = candidates.pop() {
            // Early termination: if current candidate is worse than worst result, stop
            if results.len() >= ef && current_dist > worst_dist {
                break;
            }

            // Get neighbors at this layer - copy to buffer to avoid borrow issues
            neighbor_buffer.clear();
            if layer < self.nodes[current_idx].layers.len() {
                neighbor_buffer.extend_from_slice(&self.nodes[current_idx].layers[layer]);
            }
            
            if neighbor_buffer.is_empty() {
                continue;
            }
            
            // Prefetch first neighbors for cache warmth
            for &n in neighbor_buffer.iter().take(4) {
                self.prefetch_node(n);
            }

            for &neighbor_idx in &neighbor_buffer {
                // Use bit vector for O(1) visited check
                if self.visited.insert(neighbor_idx) {
                    let dist = self.distance_to_node(query, neighbor_idx);
                    
                    // Only add if could be in top ef (fast path)
                    if results.len() < ef || dist < worst_dist {
                        candidates.push(Candidate { idx: neighbor_idx, dist });
                        results.push(ReverseCandidate { idx: neighbor_idx, dist });
                        
                        // Trim results if over capacity
                        if results.len() > ef {
                            results.pop();
                            // Update worst distance
                            if let Some(worst) = results.peek() {
                                worst_dist = worst.dist;
                            }
                        } else if dist > worst_dist {
                            worst_dist = dist;
                        }
                    }
                }
            }
        }

        // Convert to sorted vec - collect first, then sort
        let mut result_vec: Vec<(usize, f32)> = Vec::with_capacity(results.len());
        for c in results {
            result_vec.push((c.idx, c.dist));
        }
        result_vec.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Distance between query vector and node (for public API)
    #[inline]
    #[allow(dead_code)] // Convenience helper retained for future external scoring use
    fn distance(&self, query: &Vector, node_idx: usize) -> f32 {
        let node_vec = self.get_vector(node_idx);
        let dot = crate::simd::dot_product_simd(query.as_slice(), node_vec);
        1.0 - dot
    }

    /// Insert a new point into the HNSW graph
    pub fn insert(&mut self, point: Point) {
        let id_str = point.id.to_string();
        let layer = self.select_layer();

        // Initialize dimension if first insert
        if self.dim == 0 {
            self.dim = point.vector.dim();
        }

        // Add vector to contiguous storage
        self.vectors.extend_from_slice(point.vector.as_slice());

        let mut node = HnswNode {
            point: point.clone(),
            layers: vec![Vec::new(); layer + 1],
        };

        if self.nodes.is_empty() {
            self.nodes.push(node);
            let node_idx = self.nodes.len() - 1;
            self.point_id_to_index.write().insert(id_str, node_idx);
            return;
        }

        let entry_point = 0;
        let query = point.vector.as_slice();
        let mut current_layer = self.max_layers - 1;

        while current_layer > layer {
            let neighbors = self.search_layer(query, entry_point, 1, current_layer);
            if !neighbors.is_empty() {
                current_layer -= 1;
            } else {
                break;
            }
        }

        let mut candidates = if !self.nodes.is_empty() {
            self.search_layer(query, entry_point, self.ef_construction, layer)
        } else {
            Vec::new()
        };
        
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors: Vec<usize> = candidates
            .iter()
            .take(self.max_connections)
            .map(|(idx, _)| *idx)
            .collect();

        node.layers[layer] = neighbors.clone();

        self.nodes.push(node);
        let node_idx = self.nodes.len() - 1;
        self.point_id_to_index.write().insert(id_str, node_idx);

        for &neighbor_idx in &neighbors {
            if neighbor_idx < self.nodes.len() && layer < self.nodes[neighbor_idx].layers.len() {
                self.nodes[neighbor_idx].layers[layer].push(node_idx);
                if self.nodes[neighbor_idx].layers[layer].len() > self.max_connections * 2 {
                    let neighbor_vec = self.get_vector(neighbor_idx).to_vec();
                    let mut layer_connections = self.nodes[neighbor_idx].layers[layer].clone();
                    
                    layer_connections.sort_by(|&a, &b| {
                        if a < self.nodes.len() && b < self.nodes.len() {
                            let dist_a = crate::simd::l2_distance_simd(&neighbor_vec, self.get_vector(a));
                            let dist_b = crate::simd::l2_distance_simd(&neighbor_vec, self.get_vector(b));
                            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    });
                    layer_connections.truncate(self.max_connections * 2);
                    self.nodes[neighbor_idx].layers[layer] = layer_connections;
                }
            }
        }
    }

    /// Search for k nearest neighbors
    /// Optimized for speed with lower ef values
    pub fn search(&mut self, query: &Vector, k: usize, ef: Option<usize>) -> Vec<(Point, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Use ef = k * 1.5 for speed (Redis-like approach), minimum 16
        let ef = ef.unwrap_or_else(|| (k + k / 2).max(16)).max(k);
        let entry_point = 0;
        let query_slice = query.as_slice();
        
        // For small datasets, skip upper layer traversal
        if self.nodes.len() < 1000 {
            let results = self.search_layer(query_slice, entry_point, ef, 0);
            return results
                .into_iter()
                .take(k)
                .map(|(idx, dist)| {
                    let node = &self.nodes[idx];
                    let similarity = 1.0 - dist;
                    (node.point.clone(), similarity)
                })
                .collect();
        }

        let mut current_layer = self.max_layers - 1;
        while current_layer > 0 {
            let neighbors = self.search_layer(query_slice, entry_point, 1, current_layer);
            if !neighbors.is_empty() {
                current_layer -= 1;
            } else {
                break;
            }
        }

        let results = self.search_layer(query_slice, entry_point, ef, 0);
        
        results
            .into_iter()
            .take(k)
            .map(|(idx, dist)| {
                let node = &self.nodes[idx];
                let similarity = 1.0 - dist;
                (node.point.clone(), similarity)
            })
            .collect()
    }

    pub fn remove(&mut self, point_id: &str) -> bool {
        let mut index_map = self.point_id_to_index.write();
        if let Some(index) = index_map.remove(point_id) {
            // Remove from contiguous vector storage
            let start = index * self.dim;
            let end = start + self.dim;
            if end <= self.vectors.len() {
                self.vectors.drain(start..end);
            }
            
            self.nodes.remove(index);
            
            index_map.clear();
            for (i, node) in self.nodes.iter().enumerate() {
                index_map.insert(node.point.id.to_string(), i);
            }
            true
        } else {
            false
        }
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_insert_search() {
        let mut index = HnswIndex::new(16, 3);
        
        // Insert some test vectors
        for i in 0..10 {
            let vector = Vector::new(vec![i as f32, i as f32, i as f32]);
            let point = Point::new(crate::PointId::Integer(i), vector, None);
            index.insert(point);
        }

        // Search
        let query = Vector::new(vec![5.0, 5.0, 5.0]);
        let results = index.search(&query, 3, None);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_visited_set() {
        let mut vs = VisitedSet::new(100);
        
        // Test insert and contains
        assert!(!vs.contains(5));
        assert!(vs.insert(5)); // First insert returns true
        assert!(vs.contains(5));
        assert!(!vs.insert(5)); // Second insert returns false
        
        // Test clear
        vs.clear();
        assert!(!vs.contains(5));
        assert!(vs.insert(5));
    }
}
