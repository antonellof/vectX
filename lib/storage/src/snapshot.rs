// Snapshot support for persistence with LMDB
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter, Cursor};
use std::path::{Path, PathBuf};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use tar::Archive;

/// Snapshot description for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotDescription {
    pub name: String,
    pub creation_time: Option<String>,
    pub size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

/// Collection snapshot data - contains all points and config
#[derive(Debug, Serialize, Deserialize)]
pub struct CollectionSnapshotData {
    pub name: String,
    pub config: CollectionConfigData,
    pub points: Vec<PointData>,
    pub created_at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CollectionConfigData {
    pub vector_dim: usize,
    pub distance: String,
    pub use_hnsw: bool,
    pub enable_bm25: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PointData {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multivector: Option<Vec<Vec<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

pub struct SnapshotManager {
    snapshot_dir: PathBuf,
}

impl SnapshotManager {
    pub fn new<P: AsRef<Path>>(snapshot_dir: P) -> Result<Self> {
        let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
        fs::create_dir_all(&snapshot_dir)?;
        Ok(Self { snapshot_dir })
    }

    /// Get the snapshot directory for a specific collection
    fn collection_snapshot_dir(&self, collection_name: &str) -> PathBuf {
        self.snapshot_dir.join(collection_name)
    }

    /// Generate snapshot filename with timestamp
    fn generate_snapshot_name(collection_name: &str) -> String {
        let now: DateTime<Utc> = Utc::now();
        format!(
            "{}-{}.snapshot",
            collection_name,
            now.format("%Y-%m-%d-%H-%M-%S")
        )
    }

    /// Create a snapshot for a collection
    pub fn create_collection_snapshot(&self, data: CollectionSnapshotData) -> Result<SnapshotDescription> {
        let collection_dir = self.collection_snapshot_dir(&data.name);
        fs::create_dir_all(&collection_dir)?;

        let snapshot_name = Self::generate_snapshot_name(&data.name);
        let snapshot_path = collection_dir.join(&snapshot_name);

        // Serialize to JSON and compress with gzip
        let json_data = serde_json::to_vec(&data)?;
        
        let file = File::create(&snapshot_path)?;
        let mut encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
        encoder.write_all(&json_data)?;
        encoder.finish()?;

        // Calculate checksum
        let file_data = fs::read(&snapshot_path)?;
        let checksum = format!("{:x}", Sha256::digest(&file_data));

        let metadata = fs::metadata(&snapshot_path)?;
        let creation_time = metadata.created()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .and_then(|d| {
                DateTime::from_timestamp(d.as_secs() as i64, 0)
                    .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
            });

        Ok(SnapshotDescription {
            name: snapshot_name,
            creation_time,
            size: metadata.len(),
            checksum: Some(checksum),
        })
    }

    /// List all snapshots for a collection
    pub fn list_collection_snapshots(&self, collection_name: &str) -> Result<Vec<SnapshotDescription>> {
        let collection_dir = self.collection_snapshot_dir(collection_name);
        
        if !collection_dir.exists() {
            return Ok(Vec::new());
        }

        let mut snapshots = Vec::new();
        for entry in fs::read_dir(&collection_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("snapshot") {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    let metadata = fs::metadata(&path)?;
                    
                    // Calculate checksum
                    let file_data = fs::read(&path)?;
                    let checksum = format!("{:x}", Sha256::digest(&file_data));
                    
                    let creation_time = metadata.created()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .and_then(|d| {
                            DateTime::from_timestamp(d.as_secs() as i64, 0)
                                .map(|dt| dt.format("%Y-%m-%dT%H:%M:%SZ").to_string())
                        });

                    snapshots.push(SnapshotDescription {
                        name: name.to_string(),
                        creation_time,
                        size: metadata.len(),
                        checksum: Some(checksum),
                    });
                }
            }
        }

        // Sort by name (which includes timestamp)
        snapshots.sort_by(|a, b| b.name.cmp(&a.name));
        Ok(snapshots)
    }

    /// Load a snapshot from file
    pub fn load_collection_snapshot(&self, collection_name: &str, snapshot_name: &str) -> Result<CollectionSnapshotData> {
        let snapshot_path = self.collection_snapshot_dir(collection_name).join(snapshot_name);
        
        if !snapshot_path.exists() {
            return Err(anyhow!("Snapshot '{}' not found for collection '{}'", snapshot_name, collection_name));
        }

        let file = File::open(&snapshot_path)?;
        let mut decoder = GzDecoder::new(BufReader::new(file));
        let mut json_data = Vec::new();
        decoder.read_to_end(&mut json_data)?;

        let data: CollectionSnapshotData = serde_json::from_slice(&json_data)?;
        Ok(data)
    }

    /// Delete a snapshot
    pub fn delete_collection_snapshot(&self, collection_name: &str, snapshot_name: &str) -> Result<bool> {
        let snapshot_path = self.collection_snapshot_dir(collection_name).join(snapshot_name);
        
        if snapshot_path.exists() {
            fs::remove_file(&snapshot_path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get snapshot file path for download
    pub fn get_snapshot_path(&self, collection_name: &str, snapshot_name: &str) -> Option<PathBuf> {
        let path = self.collection_snapshot_dir(collection_name).join(snapshot_name);
        if path.exists() {
            Some(path)
        } else {
            None
        }
    }

    /// Download snapshot from URL and save it
    /// Supports both vectX and Qdrant snapshot formats
    pub async fn download_snapshot_from_url(
        &self,
        collection_name: &str,
        url: &str,
        expected_checksum: Option<&str>,
    ) -> Result<PathBuf> {
        let collection_dir = self.collection_snapshot_dir(collection_name);
        fs::create_dir_all(&collection_dir)?;

        // Extract filename from URL or generate one
        let filename = url
            .rsplit('/')
            .next()
            .filter(|s| s.ends_with(".snapshot"))
            .map(|s| s.to_string())
            .unwrap_or_else(|| Self::generate_snapshot_name(collection_name));

        let snapshot_path = collection_dir.join(&filename);

        // Download using reqwest
        let response = reqwest::get(url).await
            .map_err(|e| anyhow!("Failed to download snapshot: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to download snapshot: HTTP {}", response.status()));
        }

        let bytes = response.bytes().await
            .map_err(|e| anyhow!("Failed to read snapshot data: {}", e))?;

        // Verify checksum if provided
        if let Some(expected) = expected_checksum {
            let actual = format!("{:x}", Sha256::digest(&bytes));
            if actual != expected {
                return Err(anyhow!(
                    "Checksum mismatch: expected {}, got {}",
                    expected,
                    actual
                ));
            }
        }

        // Save to file
        fs::write(&snapshot_path, &bytes)?;

        Ok(snapshot_path)
    }

    /// Load snapshot from a file path (for recovery)
    /// Supports both vectX format (gzipped JSON) and Qdrant format (tar or tar.gz archive)
    pub fn load_snapshot_from_path(&self, path: &Path) -> Result<CollectionSnapshotData> {
        let file_data = fs::read(path)?;
        
        // Check if it's gzipped (magic bytes 1f 8b)
        let data = if file_data.len() > 2 && file_data[0] == 0x1f && file_data[1] == 0x8b {
            // Gzipped - decompress first
            let mut decoder = GzDecoder::new(Cursor::new(&file_data));
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed)?;
            decompressed
        } else {
            // Not gzipped - use raw data
            file_data
        };

        // Try to parse as vectX JSON format first
        if let Ok(snapshot_data) = serde_json::from_slice::<CollectionSnapshotData>(&data) {
            return Ok(snapshot_data);
        }

        // Check if it's a tar archive (Qdrant format) - "ustar" at offset 257
        if data.len() > 262 && &data[257..262] == b"ustar" {
            return self.try_parse_qdrant_snapshot(&data);
        }

        Err(anyhow!("Failed to parse snapshot: not a valid vectX or Qdrant snapshot format"))
    }

    /// Try to parse a Qdrant tar.gz snapshot and extract collection data
    fn try_parse_qdrant_snapshot(&self, tar_data: &[u8]) -> Result<CollectionSnapshotData> {
        let cursor = Cursor::new(tar_data);
        let mut archive = Archive::new(cursor);
        
        let mut collection_config: Option<serde_json::Value> = None;
        let collection_name = String::from("imported_collection");
        
        // Read through the archive looking for config.json
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_path_buf();
            let path_str = path.to_string_lossy();
            
            // Look for collection config
            if path_str.ends_with("config.json") {
                let mut content = String::new();
                entry.read_to_string(&mut content)?;
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                    collection_config = Some(config);
                }
            }
        }

        // Extract collection configuration
        let (vector_dim, distance) = if let Some(config) = &collection_config {
            let params = config.get("params").unwrap_or(config);
            
            // Try to get vector dimension and distance
            let vectors = params.get("vectors");
            let (dim, dist) = if let Some(v) = vectors {
                if let Some(size) = v.get("size").and_then(|s| s.as_u64()) {
                    let distance = v.get("distance")
                        .and_then(|d| d.as_str())
                        .unwrap_or("Cosine")
                        .to_string();
                    (size as usize, distance)
                } else {
                    // Named vectors format
                    if let Some(obj) = v.as_object() {
                        if let Some((_, first_vec)) = obj.iter().next() {
                            let size = first_vec.get("size").and_then(|s| s.as_u64()).unwrap_or(128) as usize;
                            let distance = first_vec.get("distance")
                                .and_then(|d| d.as_str())
                                .unwrap_or("Cosine")
                                .to_string();
                            (size, distance)
                        } else {
                            (128, "Cosine".to_string())
                        }
                    } else {
                        (128, "Cosine".to_string())
                    }
                }
            } else {
                (128, "Cosine".to_string())
            };

            (dim, dist)
        } else {
            return Err(anyhow!(
                "Could not find collection config in Qdrant snapshot. \
                Note: vectX can read Qdrant snapshot structure but cannot extract points from RocksDB storage. \
                To migrate data from Qdrant:\n\
                1. Run Qdrant with the snapshot restored\n\
                2. Use the scroll API to export all points\n\
                3. Import them into vectX using the upsert API"
            ));
        };

        // We found the config but cannot extract points from RocksDB
        // Return an empty collection with the right config and a helpful message
        eprintln!(
            "Note: Imported Qdrant collection config ({}D vectors, {} distance). \
            Points cannot be automatically extracted from Qdrant's RocksDB storage. \
            Please use the Qdrant scroll API to migrate points.",
            vector_dim, distance
        );

        Ok(CollectionSnapshotData {
            name: collection_name,
            config: CollectionConfigData {
                vector_dim,
                distance,
                use_hnsw: true,
                enable_bm25: false,
            },
            points: Vec::new(), // Empty - points can't be extracted from RocksDB
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        })
    }

    /// Save uploaded snapshot data to a file
    pub fn save_uploaded_snapshot(
        &self,
        collection_name: &str,
        data: &[u8],
        filename: Option<&str>,
    ) -> Result<PathBuf> {
        let collection_dir = self.collection_snapshot_dir(collection_name);
        fs::create_dir_all(&collection_dir)?;

        // Use provided filename or generate one
        let snapshot_name = filename
            .filter(|f| f.ends_with(".snapshot"))
            .map(|f| f.to_string())
            .unwrap_or_else(|| Self::generate_snapshot_name(collection_name));

        let snapshot_path = collection_dir.join(&snapshot_name);

        // Write the data directly to the file
        fs::write(&snapshot_path, data)?;

        Ok(snapshot_path)
    }

    /// List all snapshots across all collections
    pub fn list_all_snapshots(&self) -> Result<Vec<SnapshotDescription>> {
        let mut all_snapshots = Vec::new();
        
        if !self.snapshot_dir.exists() {
            return Ok(all_snapshots);
        }

        for entry in fs::read_dir(&self.snapshot_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                if let Some(collection_name) = entry.file_name().to_str() {
                    let snapshots = self.list_collection_snapshots(collection_name)?;
                    all_snapshots.extend(snapshots);
                }
            }
        }

        all_snapshots.sort_by(|a, b| b.name.cmp(&a.name));
        Ok(all_snapshots)
    }
}

