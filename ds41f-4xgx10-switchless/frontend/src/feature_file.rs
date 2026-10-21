//! Private immutable BF16 backing files. Live owners, not LRU entries, own the disk quota.
//! Protocol decoding creates an unowned path descriptor; only encoder-validated bytes create owners.
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const ROOT: &str = "/tmp/dsgb10-vision-features";
pub const MAX_LIVE_BYTES: usize = 12usize << 30;
pub const MAX_IMAGE_BYTES: usize = 1024 * 5120 * 2;
pub const MAX_IMAGES: usize = 4096;
pub const MAX_IMAGE_ROWS: usize = 1 << 20;

#[derive(Clone, Debug)]
pub struct FeatureFile {
    path: String,
    bytes: usize,
    owner: Option<Arc<OwnedFile>>,
}
impl PartialEq for FeatureFile {
    fn eq(&self, other: &Self) -> bool { self.path == other.path && self.bytes == other.bytes }
}
impl FeatureFile {
    pub fn create(bytes: &[u8]) -> Result<Self, String> { store()?.write(bytes) }
    pub fn descriptor(path: &str, bytes: usize) -> Result<Self, String> {
        validate_path(path)?;
        if bytes == 0 || bytes > MAX_IMAGE_BYTES { return Err("invalid feature file size".into()); }
        Ok(Self { path: path.into(), bytes, owner: None })
    }
    pub fn path(&self) -> &str { &self.path }
    pub fn len(&self) -> usize { self.bytes }
    pub fn is_owned(&self) -> bool { self.owner.is_some() }
    pub fn shares_owner(&self, other: &Self) -> bool {
        matches!((&self.owner, &other.owner), (Some(a),Some(b)) if Arc::ptr_eq(a,b))
    }
}

pub fn validate_path(path: &str) -> Result<(), String> {
    if path.len() > 4096 { return Err("feature path too long".into()); }
    let rest = path.strip_prefix(&format!("{ROOT}/")).ok_or("invalid feature path root")?;
    let parts: Vec<_> = rest.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|p| p.is_empty() || p.len()>128 ||
        !p.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')) {
        return Err("invalid feature path components".into());
    }
    Ok(())
}

#[derive(Debug)]
struct Store { dir: PathBuf, live: AtomicUsize, limit: usize, next: AtomicU64 }
#[derive(Debug)]
struct OwnedFile { path: PathBuf, bytes: usize, store: Arc<Store> }
impl Drop for OwnedFile {
    fn drop(&mut self) {
        // Failure keeps the quota charged. Never delete arbitrary paths or a generation recursively.
        if fs::remove_file(&self.path).is_ok() {
            self.store.live.fetch_sub(self.bytes, Ordering::AcqRel);
        }
    }
}
impl Drop for Store { fn drop(&mut self) { let _ = fs::remove_dir(&self.dir); } }

#[cfg(unix)]
fn secure_dir(path: &Path, create: bool) -> Result<(), String> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    extern "C" { fn geteuid() -> u32; }
    if create {
        let mut b=fs::DirBuilder::new(); b.mode(0o700);
        match b.create(path) { Ok(())=>(), Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>(),
            Err(e)=>return Err(format!("feature directory create failed: {e}")) }
    }
    let m=fs::symlink_metadata(path).map_err(|e|format!("feature directory unavailable: {e}"))?;
    if !m.is_dir() || m.file_type().is_symlink() || m.mode() & 0o777 != 0o700 ||
        m.uid() != unsafe { geteuid() } {
        return Err("feature directory owner/type/mode mismatch".into());
    }
    Ok(())
}
#[cfg(not(unix))]
fn secure_dir(_: &Path, _: bool) -> Result<(), String> {
    Err("feature files require the Unix engine host".into())
}

#[cfg(unix)]
fn existing_bytes(root: &Path) -> Result<usize, String> {
    use std::os::unix::fs::MetadataExt;
    extern "C" { fn geteuid() -> u32; }
    let uid = unsafe { geteuid() };
    let mut bytes = 0usize;
    let mut files = 0usize;
    let mut generations = 0usize;
    for generation in fs::read_dir(root).map_err(|e|format!("feature directory scan failed: {e}"))? {
        let generation=generation.map_err(|e|e.to_string())?;
        generations+=1;
        if generations>4096 { return Err("too many pre-existing feature generations".into()); }
        // Owners may finish while another test/store is scanning. Missing entries own no disk.
        if matches!(fs::symlink_metadata(generation.path()),Err(e) if e.kind()==std::io::ErrorKind::NotFound) {continue;}
        secure_dir(&generation.path(), false)?;
        let directory=match fs::read_dir(generation.path()) {
            Ok(d)=>d,Err(e) if e.kind()==std::io::ErrorKind::NotFound=>continue,Err(e)=>return Err(e.to_string())
        };
        for file in directory {
            let file=file.map_err(|e|e.to_string())?;
            let m=match fs::symlink_metadata(file.path()) {
                Ok(m)=>m,Err(e) if e.kind()==std::io::ErrorKind::NotFound=>continue,Err(e)=>return Err(e.to_string())
            };
            if !m.is_file() || m.file_type().is_symlink() || m.uid()!=uid ||
                !matches!(m.mode() & 0o777, 0o400 | 0o600) || m.len()>MAX_IMAGE_BYTES as u64 {
                return Err("unsafe pre-existing feature file; inspect the private spool directory".into());
            }
            files+=1;
            if files>131072 { return Err("too many pre-existing feature files".into()); }
            bytes=bytes.checked_add(m.len() as usize).ok_or("feature disk accounting overflow")?;
            if bytes>MAX_LIVE_BYTES { return Err("existing feature files exceed the 12 GiB disk budget".into()); }
        }
    }
    Ok(bytes)
}
#[cfg(not(unix))]
fn existing_bytes(_: &Path) -> Result<usize, String> { Err("Unix feature host required".into()) }

impl Store {
    fn new(limit: usize) -> Result<Arc<Self>, String> {
        secure_dir(Path::new(ROOT), true)?;
        // A previous crashed process may have left a generation behind. Charge its safe
        // files conservatively; do not unlink another generation or silently reset quota.
        let previous=existing_bytes(Path::new(ROOT))?;
        if previous>limit { return Err("pre-existing feature files exhaust the disk budget".into()); }
        let mut random=[0u8;16];
        std::fs::File::open("/dev/urandom").and_then(|mut f|f.read_exact(&mut random))
            .map_err(|e|format!("feature generation entropy unavailable: {e}"))?;
        let suffix=random.iter().map(|b|format!("{b:02x}")).collect::<String>();
        let dir=Path::new(ROOT).join(format!("g{}_{}",std::process::id(),suffix));
        // Unique generation must not pre-exist, even if its permissions happen to match.
        if fs::symlink_metadata(&dir).is_ok() { return Err("feature generation collision".into()); }
        secure_dir(&dir,true)?;
        Ok(Arc::new(Self {dir,live:AtomicUsize::new(previous),limit,next:AtomicU64::new(0)}))
    }
    fn write(self: &Arc<Self>, bytes: &[u8]) -> Result<FeatureFile,String> {
        if bytes.is_empty() || bytes.len()>MAX_IMAGE_BYTES || bytes.len()%2 != 0 {
            return Err("invalid feature file byte count".into());
        }
        if bytes.chunks_exact(2).any(|b|u16::from_le_bytes([b[0],b[1]]) & 0x7f80 == 0x7f80) {
            return Err("non-finite image embeddings".into());
        }
        self.live.fetch_update(Ordering::AcqRel,Ordering::Acquire,|n|
            n.checked_add(bytes.len()).filter(|&v|v<=self.limit))
            .map_err(|_|"active image feature files exceed the 12 GiB disk budget")?;
        let path=self.dir.join(format!("f{}",self.next.fetch_add(1,Ordering::Relaxed)));
        let text=path.to_str().ok_or("non-UTF8 feature file path")?.to_owned();
        validate_path(&text)?;
        let mut created=false;
        let mut write=||->std::io::Result<()> {
            let mut options=OpenOptions::new(); options.write(true).create_new(true);
            #[cfg(unix)] { use std::os::unix::fs::OpenOptionsExt; options.mode(0o600); }
            let mut file=options.open(&path)?;
            created=true;
            file.write_all(bytes)?;
            #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; file.set_permissions(fs::Permissions::from_mode(0o400))?; }
            file.sync_all()?;
            Ok(())
        };
        if let Err(e)=write() {
            // create_new failure must never remove somebody else's pre-existing file.
            if !created || fs::remove_file(&path).is_ok() { self.live.fetch_sub(bytes.len(),Ordering::AcqRel); }
            return Err(format!("image feature spool write failed: {e}"));
        }
        Ok(FeatureFile {path:text,bytes:bytes.len(),owner:Some(Arc::new(OwnedFile {
            path,bytes:bytes.len(),store:self.clone()
        }))})
    }
}
fn store() -> Result<Arc<Store>,String> {
    static STORE:OnceLock<Result<Arc<Store>,String>>=OnceLock::new();
    STORE.get_or_init(||Store::new(MAX_LIVE_BYTES)).clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_exact_private_paths() {
        for p in ["/tmp/dsgb10-vision-features/g/f", "/tmp/dsgb10-vision-features/g_a-1/f9"] {
            assert!(validate_path(p).is_ok());
        }
        for p in ["relative","/tmp/dsgb10-vision-features/g/../x","/tmp/dsgb10-vision-features/g/f/x",
                  "/tmp/dsgb10-vision-features//f","/tmp/dsgb10-vision-features/g/f\0","/tmp/dsgb10-vision-features/g/f.png"] {
            assert!(validate_path(p).is_err());
        }
    }
    #[cfg(unix)]
    #[test]
    fn immutable_owner_and_live_quota_survive_cache_eviction() {
        use std::os::unix::fs::MetadataExt;
        let mut s=Store::new(MAX_LIVE_BYTES).unwrap();
        let baseline=s.live.load(Ordering::Acquire);
        Arc::get_mut(&mut s).unwrap().limit=baseline+64;
        let a=s.write(&[0;32]).unwrap(); let p=a.path().to_owned();let b=a.clone();
        assert_eq!(fs::metadata(&p).unwrap().mode()&0o777,0o400);
        assert_eq!(fs::read(&p).unwrap(),vec![0;32]);
        drop(a);assert!(Path::new(&p).exists());assert_eq!(s.live.load(Ordering::Acquire),baseline+32);
        assert!(s.write(&[0;34]).is_err());assert_eq!(s.live.load(Ordering::Acquire),baseline+32);
        drop(b);assert!(!Path::new(&p).exists());assert_eq!(s.live.load(Ordering::Acquire),baseline);
        assert!(s.write(&[0x80,0x7f]).is_err());assert_eq!(s.live.load(Ordering::Acquire),baseline);
    }
    #[cfg(unix)]
    #[test]
    fn collision_preserves_existing_file_and_releases_reserved_quota() {
        use std::os::unix::fs::OpenOptionsExt;
        let s=Store::new(MAX_LIVE_BYTES).unwrap();
        let baseline=s.live.load(Ordering::Acquire);
        let path=s.dir.join("f0");
        OpenOptions::new().create_new(true).write(true).mode(0o600).open(&path).unwrap().write_all(b"existing").unwrap();
        assert!(s.write(&[0;32]).is_err());
        assert_eq!(fs::read(&path).unwrap(),b"existing");
        assert_eq!(s.live.load(Ordering::Acquire),baseline);
        fs::remove_file(path).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn startup_counts_old_files_without_following_links() {
        use std::os::unix::fs::{OpenOptionsExt,symlink};
        let test_root=std::env::temp_dir().join(format!("feature-scan-test-{}-{}",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        secure_dir(&test_root,true).unwrap();
        let old=test_root.join("old");secure_dir(&old,true).unwrap();
        let file=old.join("f0");
        OpenOptions::new().create_new(true).write(true).mode(0o600).open(&file).unwrap().write_all(&[0;32]).unwrap();
        assert_eq!(existing_bytes(&test_root).unwrap(),32);
        let link=old.join("link");symlink("/etc/passwd",&link).unwrap();
        assert!(existing_bytes(&test_root).is_err());
        fs::remove_file(link).unwrap();fs::remove_file(file).unwrap();
        fs::remove_dir(old).unwrap();fs::remove_dir(test_root).unwrap();
    }
}
