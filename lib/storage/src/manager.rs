use vectx_core::{Collection, CollectionConfig, Distance, Error, Result, Point, PointId, Vector, MultiVector};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use crate::lmdb_storage::LmdbStorage;
use crate::wal::WriteAheadLog;
use crate::snapshot::{SnapshotManager, SnapshotDescription, CollectionSnapshotData, CollectionConfigData, PointData};
use crate::persistence::ForkBasedPersistence;

/// Manages collections and persistence
pub struct StorageManager {
    collections: Arc<RwLock<HashMap<String, Arc<Collection>>>>,
    /// Aliases: alias_name -> collection_name
    aliases: Arc<RwLock<HashMap<String, String>>>,
    data_dir: PathBuf,
    #[allow(dead_code)]
    lmdb: Option<Arc<LmdbStorage>>,
    #[allow(dead_code)]
    wal: Option<Arc<WriteAheadLog>>,
    snapshots: Arc<SnapshotManager>,
    persistence: Arc<ForkBasedPersistence>,
    #[allow(dead_code)]
    save_interval: Option<Duration>,
}

impl StorageManager {
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        let lmdb_path = data_dir.join("lmdb");
        let lmdb = Arc::new(LmdbStorage::new(&lmdb_path)
            .map_err(|e| Error::Storage(e.to_string()))?);

        let wal_path = data_dir.join("wal.log");
        let wal = Arc::new(WriteAheadLog::new(&wal_path)
            .map_err(|e| Error::Storage(e.to_string()))?);

        let snapshot_dir = data_dir.join("snapshots");
        let snapshots = Arc::new(SnapshotManager::new(&snapshot_dir)
            .map_err(|e| Error::Storage(e.to_string()))?);

        let persistence = Arc::new(ForkBasedPersistence::new(&data_dir));

        let collections = Arc::new(RwLock::new(HashMap::new()));
        let aliases = Arc::new(RwLock::new(HashMap::new()));
        
        if let Some(snapshot) = persistence.load_snapshot()
            .map_err(|e| Error::Persistence(e.to_string()))? {
            eprintln!("Loading snapshot from disk...");
            let mut collections_map = HashMap::new();
            
            for col_snapshot in snapshot.collections {
                let config = CollectionConfig {
                    name: col_snapshot.name.clone(),
                    vector_dim: col_snapshot.config.vector_dim,
                    distance: match col_snapshot.config.distance.as_str() {
                        "Cosine" => Distance::Cosine,
                        "Euclidean" => Distance::Euclidean,
                        "Dot" => Distance::Dot,
                        _ => Distance::Cosine,
                    },
                    use_hnsw: col_snapshot.config.use_hnsw,
                    enable_bm25: col_snapshot.config.enable_bm25,
                };
                
                let collection = Arc::new(Collection::new(config));
                
                for point_snapshot in col_snapshot.points {
                    let point = Point::new(
                        PointId::String(point_snapshot.id.clone()),
                        Vector::new(point_snapshot.vector),
                        point_snapshot.payload,
                    );
                    if let Err(e) = collection.upsert(point) {
                        eprintln!("Warning: Failed to restore point {}: {}", point_snapshot.id, e);
                    }
                }
                
                collections_map.insert(col_snapshot.name, collection);
            }
            
            *collections.write() = collections_map;
            eprintln!("Snapshot loaded: {} collections", collections.read().len());
        }

        let manager = Self {
            collections,
            aliases,
            data_dir,
            lmdb: Some(lmdb),
            wal: Some(wal),
            snapshots,
            persistence,
            save_interval: Some(Duration::from_secs(300)),
        };

        manager.start_background_save();

        Ok(manager)
    }

    /// Start background save thread
    fn start_background_save(&self) {
        let collections = self.collections.clone();
        let persistence = self.persistence.clone();
        let interval = self.save_interval.unwrap_or(Duration::from_secs(300));

        std::thread::spawn(move || {
            loop {
                std::thread::sleep(interval);
                
                if !ForkBasedPersistence::is_bgsave_in_progress() {
                    let collections_map = collections.read();
                    if let Err(e) = persistence.bgsave(&collections_map) {
                        eprintln!("Background save error: {}", e);
                    }
                }
            }
        });
    }

    pub fn create_collection(&self, config: CollectionConfig) -> Result<Arc<Collection>> {
        let name = config.name.clone();
        let mut collections = self.collections.write();

        if collections.contains_key(&name) {
            return Err(Error::CollectionExists(name));
        }

        let collection = Arc::new(Collection::new(config));
        collections.insert(name.clone(), collection.clone());
        Ok(collection)
    }

    #[inline]
    pub fn get_collection(&self, name: &str) -> Option<Arc<Collection>> {
        let collections = self.collections.read();
        // First try direct collection lookup
        if let Some(col) = collections.get(name) {
            return Some(col.clone());
        }
        // Then try alias lookup
        let aliases = self.aliases.read();
        if let Some(collection_name) = aliases.get(name) {
            return collections.get(collection_name).cloned();
        }
        None
    }

    pub fn delete_collection(&self, name: &str) -> Result<bool> {
        let mut collections = self.collections.write();
        let removed = collections.remove(name).is_some();
        Ok(removed)
    }

    #[inline]
    #[must_use]
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    #[inline]
    #[must_use]
    pub fn collection_exists(&self, name: &str) -> bool {
        self.collections.read().contains_key(name)
    }

    /// Create an alias for a collection
    pub fn create_alias(&self, alias_name: &str, collection_name: &str) -> Result<bool> {
        // Check that collection exists
        if !self.collection_exists(collection_name) {
            return Err(Error::CollectionNotFound(collection_name.to_string()));
        }
        let mut aliases = self.aliases.write();
        aliases.insert(alias_name.to_string(), collection_name.to_string());
        Ok(true)
    }

    /// Delete an alias
    pub fn delete_alias(&self, alias_name: &str) -> Result<bool> {
        let mut aliases = self.aliases.write();
        Ok(aliases.remove(alias_name).is_some())
    }

    /// Rename an alias
    pub fn rename_alias(&self, old_alias: &str, new_alias: &str) -> Result<bool> {
        let mut aliases = self.aliases.write();
        if let Some(collection_name) = aliases.remove(old_alias) {
            aliases.insert(new_alias.to_string(), collection_name);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all aliases
    pub fn list_aliases(&self) -> Vec<(String, String)> {
        self.aliases.read()
            .iter()
            .map(|(alias, collection)| (alias.clone(), collection.clone()))
            .collect()
    }

    /// List aliases for a specific collection
    pub fn list_collection_aliases(&self, collection_name: &str) -> Vec<String> {
        self.aliases.read()
            .iter()
            .filter(|(_, col)| *col == collection_name)
            .map(|(alias, _)| alias.clone())
            .collect()
    }

    #[inline]
    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }


    /// Trigger background save
    pub fn bgsave(&self) -> Result<bool> {
        let collections = self.collections.read();
        self.persistence.bgsave(&collections)
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Force save
    pub fn save(&self) -> Result<()> {
        let collections = self.collections.read();
        self.persistence.save(&collections)
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Get last save time
    pub fn last_save_time(&self) -> u64 {
        ForkBasedPersistence::last_save_time()
    }

    /// Check if background save is in progress
    pub fn is_bgsave_in_progress(&self) -> bool {
        ForkBasedPersistence::is_bgsave_in_progress()
    }

    // ==================== Snapshot Methods ====================

    /// Create a snapshot for a collection
    pub fn create_collection_snapshot(&self, collection_name: &str) -> Result<SnapshotDescription> {
        let collections = self.collections.read();
        let collection = collections.get(collection_name)
            .ok_or_else(|| Error::CollectionNotFound(collection_name.to_string()))?;

        let points = collection.get_all_points();

        let snapshot_data = CollectionSnapshotData {
            name: collection_name.to_string(),
            config: CollectionConfigData {
                vector_dim: collection.vector_dim(),
                distance: match collection.distance() {
                    Distance::Cosine => "Cosine".to_string(),
                    Distance::Euclidean => "Euclidean".to_string(),
                    Distance::Dot => "Dot".to_string(),
                },
                use_hnsw: collection.use_hnsw(),
                enable_bm25: collection.enable_bm25(),
            },
            points: points.iter().map(|p| PointData {
                id: match &p.id {
                    PointId::Integer(i) => i.to_string(),
                    PointId::String(s) => s.clone(),
                    PointId::Uuid(u) => u.to_string(),
                },
                vector: p.vector.as_slice().to_vec(),
                multivector: p.multivector.as_ref().map(|mv: &MultiVector| mv.vectors().to_vec()),
                payload: p.payload.clone(),
            }).collect(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };

        self.snapshots.create_collection_snapshot(snapshot_data)
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// List snapshots for a collection
    pub fn list_collection_snapshots(&self, collection_name: &str) -> Result<Vec<SnapshotDescription>> {
        self.snapshots.list_collection_snapshots(collection_name)
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Delete a snapshot
    pub fn delete_collection_snapshot(&self, collection_name: &str, snapshot_name: &str) -> Result<bool> {
        self.snapshots.delete_collection_snapshot(collection_name, snapshot_name)
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Get snapshot file path for download
    pub fn get_snapshot_path(&self, collection_name: &str, snapshot_name: &str) -> Option<PathBuf> {
        self.snapshots.get_snapshot_path(collection_name, snapshot_name)
    }

    /// Recover collection from a snapshot file
    pub fn recover_from_snapshot(&self, collection_name: &str, snapshot_name: &str) -> Result<Arc<Collection>> {
        let snapshot_data = self.snapshots.load_collection_snapshot(collection_name, snapshot_name)
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.restore_collection_from_data_with_name(snapshot_data, Some(collection_name))
    }

    /// Recover collection from a URL
    pub async fn recover_from_url(&self, collection_name: &str, url: &str, checksum: Option<&str>) -> Result<Arc<Collection>> {
        let snapshot_path = self.snapshots.download_snapshot_from_url(collection_name, url, checksum)
            .await
            .map_err(|e| Error::Storage(e.to_string()))?;

        let snapshot_data = self.snapshots.load_snapshot_from_path(&snapshot_path)
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.restore_collection_from_data_with_name(snapshot_data, Some(collection_name))
    }

    fn restore_collection_from_data_with_name(&self, data: CollectionSnapshotData, target_name: Option<&str>) -> Result<Arc<Collection>> {
        let collection_name = target_name.unwrap_or(&data.name).to_string();
        
        let config = CollectionConfig {
            name: collection_name.clone(),
            vector_dim: data.config.vector_dim,
            distance: match data.config.distance.as_str() {
                "Cosine" => Distance::Cosine,
                "Euclidean" => Distance::Euclidean,
                "Dot" => Distance::Dot,
                _ => Distance::Cosine,
            },
            use_hnsw: data.config.use_hnsw,
            enable_bm25: data.config.enable_bm25,
        };

        {
            let mut collections = self.collections.write();
            collections.remove(&collection_name);
        }

        let collection = Arc::new(Collection::new(config));

        for point_data in data.points {
            let point_id = point_data.id.parse::<u64>()
                .map(PointId::Integer)
                .unwrap_or_else(|_| PointId::String(point_data.id.clone()));

            let point = if let Some(mv_data) = point_data.multivector {
                match MultiVector::new(mv_data) {
                    Ok(mv) => Point::new_multi(point_id, mv, point_data.payload),
                    Err(e) => {
                        eprintln!("Warning: Failed to create multivector: {}", e);
                        Point::new(point_id, Vector::new(point_data.vector), point_data.payload)
                    }
                }
            } else {
                Point::new(
                    point_id,
                    Vector::new(point_data.vector),
                    point_data.payload,
                )
            };

            if let Err(e) = collection.upsert(point) {
                eprintln!("Warning: Failed to restore point: {}", e);
            }
        }

        {
            let mut collections = self.collections.write();
            collections.insert(collection_name, collection.clone());
        }

        Ok(collection)
    }

    /// List all snapshots
    pub fn list_all_snapshots(&self) -> Result<Vec<SnapshotDescription>> {
        self.snapshots.list_all_snapshots()
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Upload and restore a snapshot from raw bytes
    pub fn upload_and_restore_snapshot(
        &self, 
        collection_name: &str, 
        data: &[u8],
        filename: Option<&str>,
    ) -> Result<Arc<Collection>> {
        let snapshot_path = self.snapshots.save_uploaded_snapshot(collection_name, data, filename)
            .map_err(|e| Error::Storage(e.to_string()))?;

        let snapshot_data = self.snapshots.load_snapshot_from_path(&snapshot_path)
            .map_err(|e| Error::Storage(e.to_string()))?;

        self.restore_collection_from_data_with_name(snapshot_data, Some(collection_name))
    }
}
