// Integration tests for vectX
use vectx_core::{Collection, CollectionConfig, Distance, Point, PointId, Vector};
use vectx_storage::StorageManager;

#[test]
fn test_collection_creation() {
    let config = CollectionConfig {
        name: "test_collection".to_string(),
        vector_dim: 128,
        distance: Distance::Cosine,
        use_hnsw: true,
        enable_bm25: false,
    };
    
    let collection = Collection::new(config);
    assert_eq!(collection.name(), "test_collection");
    assert_eq!(collection.vector_dim(), 128);
    assert_eq!(collection.count(), 0);
}

#[test]
fn test_point_insertion() {
    let config = CollectionConfig {
        name: "test".to_string(),
        vector_dim: 3,
        distance: Distance::Cosine,
        use_hnsw: false,
        enable_bm25: false,
    };
    
    let collection = Collection::new(config);
    
    let point = Point::new(
        PointId::String("point1".to_string()),
        Vector::new(vec![1.0, 2.0, 3.0]),
        Some(serde_json::json!({"name": "test"})),
    );
    
    assert!(collection.upsert(point).is_ok());
    assert_eq!(collection.count(), 1);
    
    let retrieved = collection.get("point1");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().vector.dim(), 3);
}

#[test]
fn test_vector_search() {
    let config = CollectionConfig {
        name: "test".to_string(),
        vector_dim: 3,
        distance: Distance::Cosine,
        use_hnsw: false, // Use linear search for small dataset
        enable_bm25: false,
    };
    
    let collection = Collection::new(config);
    
    // Insert some points
    for i in 0..10 {
        let vector = Vector::new(vec![i as f32, i as f32, i as f32]);
        let point = Point::new(
            PointId::Integer(i),
            vector,
            None,
        );
        collection.upsert(point).unwrap();
    }
    
    // Search for similar vector
    let query = Vector::new(vec![5.0, 5.0, 5.0]);
    let results = collection.search(&query, 3, None);
    
    assert_eq!(results.len(), 3);
    // Should find points close to [5,5,5]
    assert!(results[0].1 > 0.9); // High similarity
}

#[test]
fn test_bm25_search() {
    let config = CollectionConfig {
        name: "test".to_string(),
        vector_dim: 3,
        distance: Distance::Cosine,
        use_hnsw: false,
        enable_bm25: true,
    };
    
    let collection = Collection::new(config);
    
    // Insert documents with text
    for i in 0..10 {
        let point = Point::new(
            PointId::String(format!("doc{}", i)),
            Vector::new(vec![0.0, 0.0, 0.0]),
            Some(serde_json::json!({
                "text": format!("document about topic {}", i)
            })),
        );
        collection.upsert(point).unwrap();
    }
    
    // Search
    let results = collection.search_text("topic", 5);
    assert!(!results.is_empty());
    assert!(results[0].1 >= 0.0); // Should have a score (can be 0.0)
}

#[test]
fn test_storage_manager() {
    let temp_dir = tempfile::tempdir().unwrap();
    let storage = StorageManager::new(temp_dir.path()).unwrap();
    
    let config = CollectionConfig {
        name: "test_collection".to_string(),
        vector_dim: 128,
        distance: Distance::Cosine,
        use_hnsw: true,
        enable_bm25: false,
    };
    
    let collection = storage.create_collection(config).unwrap();
    assert_eq!(collection.name(), "test_collection");
    
    let retrieved = storage.get_collection("test_collection");
    assert!(retrieved.is_some());
    
    let collections = storage.list_collections();
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0], "test_collection");
}

#[test]
fn test_persistence_snapshot() {
    // Use unique temp directory for each test to avoid LMDB conflicts
    let temp_dir = tempfile::tempdir().unwrap();
    let storage = StorageManager::new(temp_dir.path()).unwrap();
    
    // Create collection and insert data
    let config = CollectionConfig {
        name: "persistent".to_string(),
        vector_dim: 3,
        distance: Distance::Cosine,
        use_hnsw: false,
        enable_bm25: false,
    };
    
    let collection = storage.create_collection(config).unwrap();
    
    for i in 0..10 {
        let point = Point::new(
            PointId::Integer(i),
            Vector::new(vec![i as f32, i as f32, i as f32]),
            None,
        );
        collection.upsert(point).unwrap();
    }
    
    // Force save
    storage.save().unwrap();
    
    // Drop the first storage manager to close LMDB environment
    drop(storage);
    
    // Create new storage manager (simulates restart)
    let storage2 = StorageManager::new(temp_dir.path()).unwrap();
    let restored = storage2.get_collection("persistent");
    
    assert!(restored.is_some());
    assert_eq!(restored.unwrap().count(), 10);
}

#[test]
fn test_payload_filtering() {
    let config = CollectionConfig {
        name: "test".to_string(),
        vector_dim: 3,
        distance: Distance::Cosine,
        use_hnsw: false,
        enable_bm25: false,
    };
    
    let collection = Collection::new(config);
    
    // Insert points with different categories
    for i in 0..20 {
        let point = Point::new(
            PointId::Integer(i),
            Vector::new(vec![i as f32, i as f32, i as f32]),
            Some(serde_json::json!({
                "category": if i % 2 == 0 { "A" } else { "B" },
                "score": i
            })),
        );
        collection.upsert(point).unwrap();
    }
    
    // Search with filter
    use vectx_core::{PayloadFilter, FilterCondition};
    let query = Vector::new(vec![10.0, 10.0, 10.0]);
    let filter = PayloadFilter::new(FilterCondition::Equals {
        field: "category".to_string(),
        value: serde_json::json!("A"),
    });
    
    let results = collection.search(&query, 10, Some(&filter));
    
    // All results should have category "A"
    for (point, _) in results {
        let category = point.payload
            .as_ref()
            .and_then(|p| p.get("category"))
            .and_then(|v| v.as_str());
        assert_eq!(category, Some("A"));
    }
}

// ==================== Similarity Engine Tests ====================

#[test]
fn test_vector_search_with_collection() {
    // Create collection
    let config = CollectionConfig {
        name: "products".to_string(),
        vector_dim: 3,  // Simple mock vectors
        distance: Distance::Cosine,
        use_hnsw: false,
        enable_bm25: false,
    };
    
    let collection = Collection::new(config);
    
    // Insert products with mock vectors
    let products = [(vec![1.0, 0.0, 0.0], serde_json::json!({"name": "Prosciutto cotto", "price": 1.99, "category": "salumi"})),
        (vec![0.9, 0.1, 0.0], serde_json::json!({"name": "Prosciutto crudo", "price": 2.49, "category": "salumi"})),
        (vec![0.8, 0.2, 0.0], serde_json::json!({"name": "Mortadella", "price": 1.79, "category": "salumi"})),
        (vec![0.0, 0.0, 1.0], serde_json::json!({"name": "iPhone 15", "price": 999.0, "category": "electronics"})),
        (vec![0.1, 0.0, 0.9], serde_json::json!({"name": "Samsung Galaxy", "price": 899.0, "category": "electronics"}))];
    
    for (i, (vec, payload)) in products.iter().enumerate() {
        let point = Point::new(
            PointId::Integer(i as u64),
            Vector::new(vec.clone()),
            Some(payload.clone()),
        );
        collection.upsert(point).unwrap();
    }
    
    // Query with a vector similar to salumi products
    let query_vector = Vector::new(vec![0.95, 0.05, 0.0]);
    
    let results = collection.search(&query_vector, 3, None);
    assert_eq!(results.len(), 3);
    
    // Top results should be salumi products (similar vectors)
    let top_category = results[0].0.payload
        .as_ref()
        .and_then(|p| p.get("category"))
        .and_then(|v| v.as_str());
    assert_eq!(top_category, Some("salumi"));
}

