//! Private, bounded connection to the same checkpoint's native vision encoder.
//! Its output is a complete BF16 image span, never a caption or text substitution.
use crate::encoding::IMAGE_PLACEHOLDER;
use crate::protocol::ImageSpan;
use crate::tokenizer::Tokenizer;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use std::time::Duration;
use crate::feature_file::{FeatureFile, MAX_IMAGES, MAX_IMAGE_ROWS};
use std::collections::HashMap;

pub const IMAGE_ID: u32 = 129264;
// Historical features are file-backed. This is a transient encoder memory limit, not a
// sum over all historical image spans; each request retains every original image.
pub const MAX_FEATURE_BYTES: usize = 512 << 20;
pub const MAX_REQUEST_BYTES: usize = 256 << 20;
pub const DIM: usize = 5120;

pub struct Expanded {
    pub prompt: String,
    pub scope: String,
    pub images: Vec<ImageSpan>,
    pub cache_images: Vec<crate::session::ImageKey>,
}

fn digest(s: &str) -> Result<[u8; 32], String> {
    if s.len() != 64 || !s.bytes().all(|x| x.is_ascii_hexdigit()) {
        return Err("invalid vision identity".into());
    }
    let mut out = [0; 32];
    for (i, x) in out.iter_mut().enumerate() {
        *x = u8::from_str_radix(&s[i*2..i*2+2], 16).map_err(|_| "invalid vision identity")?;
    }
    Ok(out)
}

/// Expand placeholders as real repeated image token spellings. This retains the
/// tokenizer/session table's exact byte-to-token invariant even at image boundaries.
pub fn expand(prompt: &str, namespace: &str, tok: &Tokenizer, threads: usize,
              identity: &str, meta: &[Value], blobs: Vec<Vec<u8>>) -> Result<Expanded, String> {
    if meta.is_empty() || meta.len()>MAX_IMAGES || meta.len()!=blobs.len() { return Err("bad native vision image count".into()); }
    let files:Result<Vec<_>,String> = blobs.into_iter().enumerate().map(|(i,b)| {
        let rows=meta[i]["rows"].as_u64().filter(|r|(1..=1024).contains(r)).ok_or("invalid image row count")? as usize;
        if meta[i]["dim"].as_u64()!=Some(DIM as u64) || b.len()!=rows*DIM*2 { return Err("invalid image embedding dimensions".into()); }
        FeatureFile::create(&b)
    }).collect();
    expand_shared(prompt, namespace, tok, threads, identity, meta, files?)
}

fn expand_shared(prompt: &str, namespace: &str, tok: &Tokenizer, threads: usize,
                 identity: &str, meta: &[Value], blobs: Vec<FeatureFile>) -> Result<Expanded, String> {
    digest(identity)?;
    if meta.is_empty() || meta.len() > MAX_IMAGES || meta.len() != blobs.len() {
        return Err("bad native vision image count".into());
    }
    let spelling = std::str::from_utf8(tok.token_bytes(IMAGE_ID)).map_err(|_| "invalid image token spelling")?;
    if tok.token_id(spelling) != Some(IMAGE_ID) { return Err("image tokenizer mismatch".into()); }
    let parts: Vec<_> = prompt.split(IMAGE_PLACEHOLDER).collect();
    if parts.len() != meta.len() + 1 { return Err("image placeholder count mismatch".into()); }
    // Reject literal image-token spellings outside image blocks, including alternate tokenizer spelling.
    if parts.iter().any(|p| tok.encode_parallel(p, threads).contains(&IMAGE_ID)) {
        return Err("image tokens must come from image content blocks".into());
    }
    let mut result = Expanded { prompt: String::new(), scope: format!("{namespace}:vision-prefix-v2:{identity}"), images: Vec::new(), cache_images: Vec::new() };
    let mut count = 0;
    let mut image_rows = 0usize;
    for (i, blob) in blobs.into_iter().enumerate() {
        let rows = meta[i]["rows"].as_u64().filter(|n| (1..=1024).contains(n)).ok_or("invalid image row count")? as usize;
        if meta[i]["dim"].as_u64() != Some(DIM as u64) || blob.len() != rows * DIM * 2 {
            return Err("invalid image embedding dimensions".into());
        }
        image_rows=image_rows.checked_add(rows).ok_or("image row count overflow")?;
        if image_rows>MAX_IMAGE_ROWS { return Err("historical image rows exceed the 1M context budget".into()); }
        let hash = meta[i]["digest"].as_str().ok_or("missing image identity")?;
        let hash_bytes = digest(hash)?;
        result.prompt.push_str(parts[i]);
        count += tok.encode_parallel(parts[i], threads).len();
        result.cache_images.push(crate::session::ImageKey {
            start: result.prompt.len(), len: rows * spelling.len(), digest: hash_bytes });
        result.images.push(ImageSpan { start: count as u32, rows: rows as u32, digest: hash_bytes, feature: blob });
        for _ in 0..rows { result.prompt.push_str(spelling); }
        count += rows;
    }
    result.prompt.push_str(parts[meta.len()]);
    Ok(result)
}


#[derive(Clone)]
struct Feature { meta: Value, blob: FeatureFile, used: u64, cost: usize }
pub struct FeatureCache {
    identity: String, entries: HashMap<String,Feature>, bytes: usize, clock: u64, limit: usize,
}
impl Default for FeatureCache {
    // This is retained file-data cost, not resident BF16 memory. Pending owners survive LRU eviction.
    fn default() -> Self { Self { identity:String::new(), entries:HashMap::new(), bytes:0, clock:0, limit:2*MAX_FEATURE_BYTES } }
}
impl FeatureCache {
    fn bind(&mut self, identity:&str) {
        if self.identity != identity { self.entries.clear();self.bytes=0;self.identity=identity.into(); }
    }
    fn get(&mut self,key:&str) -> Option<(Value,FeatureFile)> {
        self.clock+=1;
        self.entries.get_mut(key).map(|x| {x.used=self.clock;(x.meta.clone(),x.blob.clone())})
    }
    fn put(&mut self,key:String,meta:Value,blob:FeatureFile) {
        let cost=key.len()+meta.to_string().len()+blob.len()+256;
        if cost>self.limit{return;}
        if let Some(old)=self.entries.remove(&key){self.bytes-=old.cost;}
        while self.bytes+cost>self.limit {
            let Some(victim)=self.entries.iter().min_by_key(|(_,v)|v.used).map(|(k,_)|k.clone()) else {break;};
            self.bytes-=self.entries.remove(&victim).unwrap().cost;
        }
        self.clock+=1;self.bytes+=cost;
        self.entries.insert(key,Feature {meta,blob,used:self.clock,cost});
    }
}

// Only exact, immutable inline image records are reusable without asking the
// encoder again. HTTP URLs may change in place and must always be fetched afresh.
fn immutable_key(record:&Value) -> Option<String> {
    let source=record.get("source").unwrap_or(&Value::Null);
    let data=record.get("data").or_else(||source.get("data"));
    let url=record.get("url").or_else(||source.get("url"));
    if data.and_then(Value::as_str).is_some() || url.and_then(Value::as_str).map_or(false,|s|s.starts_with("data:image/")) {
        let mut hash=crate::sha256::Sha256::new();
        serde_json::to_writer(&mut hash,record).ok()?;
        Some(hash.finish())
    } else {None}
}

async fn exchange(socket:&str, revision:&str, body:Value, image_count:usize) -> Result<(Value,Vec<FeatureFile>),String> {
    if image_count>8 { return Err("encoder batch exceeds 80 MiB feature budget".into()); }
    let bytes=serde_json::to_vec(&body).map_err(|e|e.to_string())?;
    if bytes.len()>MAX_REQUEST_BYTES{return Err("combined image payload exceeds 256 MiB HTTP request budget".into());}
    let work=async {
        let mut stream=UnixStream::connect(socket).await.map_err(|_|"native vision encoder unavailable")?;
        stream.write_u32_le(bytes.len() as u32).await.map_err(|_|"vision request write failed")?;
        stream.write_all(&bytes).await.map_err(|_|"vision request write failed")?;
        let size=stream.read_u32_le().await.map_err(|_|"vision header missing")? as usize;
        if size>image_count.saturating_mul(1024).saturating_add(4096).min(MAX_REQUEST_BYTES){return Err("vision header too large".into());}
        let mut b=vec![0;size];stream.read_exact(&mut b).await.map_err(|_|"vision header truncated")?;
        let h:Value=serde_json::from_slice(&b).map_err(|_|"invalid vision header")?;
        if h["ok"]!=true{return Err("native image decoding/encoding failed".into());}
        if h["revision"].as_str()!=Some(revision){return Err("vision and language checkpoint revisions differ".into());}
        digest(h["identity"].as_str().ok_or("missing encoder identity")?)?;
        let mut blobs=Vec::new();let mut total=0;
        if image_count>0 {
            let meta=h["images"].as_array().ok_or("missing image descriptors")?;
            if meta.len()!=image_count{return Err("encoder returned wrong image count".into());}
            for m in meta {
                let rows=m["rows"].as_u64().filter(|n|(1..=1024).contains(n)).ok_or("bad image span length")? as usize;
                if m["dim"].as_u64()!=Some(DIM as u64){return Err("bad image dimension".into());}
                digest(m["digest"].as_str().ok_or("missing image identity")?)?;
                total+=rows*DIM*2;if total>8*crate::feature_file::MAX_IMAGE_BYTES{return Err("encoder feature batch exceeds 80 MiB".into());}
                let mut blob=vec![0;rows*DIM*2];stream.read_exact(&mut blob).await.map_err(|_|"image embedding payload truncated")?;
                if blob.chunks_exact(2).any(|b|u16::from_le_bytes([b[0],b[1]])&0x7f80==0x7f80){return Err("non-finite image embeddings".into());}
                blobs.push(FeatureFile::create(&blob)?);
            }
        }
        Ok((h,blobs))
    };
    tokio::time::timeout(Duration::from_secs(180),work).await.map_err(|_|"native vision encoder timed out".to_string())?
}

pub async fn materialize_cached(cache:&mut FeatureCache, socket:&str, revision:&str, records:&[Value], prompt:&str,
                         namespace:&str,tok:&Tokenizer,threads:usize)->Result<Expanded,String> {
    let (health,_)=exchange(socket,revision,json!({"op":"health"}),0).await?;
    let identity=health["identity"].as_str().unwrap();cache.bind(identity);
    if records.is_empty(){return Ok(Expanded {prompt:prompt.into(),scope:format!("{namespace}:vision-prefix-v2:{identity}"),images:Vec::new(),cache_images:Vec::new()});}
    if records.len()>MAX_IMAGES { return Err("image file descriptor budget exceeded (4096)".into()); }
    let keys:Vec<_>=records.iter().map(immutable_key).collect();
    let mut features:Vec<Option<(Value,FeatureFile)>>=keys.iter().map(|k|k.as_ref().and_then(|k|cache.get(k))).collect();
    let mut total_rows=0usize;
    for (meta,_) in features.iter().flatten() {
        let rows=meta["rows"].as_u64().filter(|n|(1..=1024).contains(n)).ok_or("invalid cached image row count")? as usize;
        total_rows=total_rows.checked_add(rows).ok_or("image row overflow")?;
    }
    if total_rows>MAX_IMAGE_ROWS { return Err("historical image rows exceed the 1M context budget".into()); }
    // Collapse exact immutable records before asking the encoder; every occurrence retains its own span.
    let mut groups:Vec<Vec<usize>>=Vec::new();let mut seen:HashMap<&str,usize>=HashMap::new();
    for (i,key) in keys.iter().enumerate() {
        if features[i].is_some(){continue;}
        if let Some(key)=key {
            if let Some(&g)=seen.get(key.as_str()){groups[g].push(i);continue;}
            seen.insert(key,groups.len());
        }
        groups.push(vec![i]);
    }
    for batch in groups.chunks(8) {
        let fresh:Vec<_>=batch.iter().map(|g|records[g[0]].clone()).collect();
        let (header,blobs)=exchange(socket,revision,json!({"images":fresh}),batch.len()).await?;
        if header["identity"].as_str()!=Some(identity){cache.entries.clear();cache.bytes=0;return Err("vision identity changed while preparing request; retry required".into());}
        let meta=header["images"].as_array().unwrap();
        for ((group,m),b) in batch.iter().zip(meta).zip(blobs) {
            total_rows=total_rows.checked_add(group.len()*m["rows"].as_u64().unwrap() as usize).ok_or("image row overflow")?;
            if total_rows>MAX_IMAGE_ROWS { return Err("historical image rows exceed the 1M context budget".into()); }
            if let Some(k)=&keys[group[0]]{cache.put(k.clone(),m.clone(),b.clone());}
            for &i in group {features[i]=Some((m.clone(),b.clone()));}
        }
    }
    let (meta,blobs):(Vec<_>,Vec<_>)=features.into_iter().map(Option::unwrap).unzip();
    // Pending/cache now pin immutable files, never the whole historical BF16 allocation.
    expand_shared(prompt,namespace,tok,threads,identity,&meta,blobs)

}

pub async fn materialize(socket:&str,revision:&str,records:&[Value],prompt:&str,namespace:&str,tok:&Tokenizer,threads:usize)->Result<Expanded,String>{
    materialize_cached(&mut FeatureCache::default(),socket,revision,records,prompt,namespace,tok,threads).await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immutable_keys_do_not_cache_mutable_urls_or_paths() {
        assert!(immutable_key(&json!({"url":"https://test.invalid/image.png"})).is_none());
        assert!(immutable_key(&json!({"source":{"url":"http://test.invalid/image.png"}})).is_none());
        assert!(immutable_key(&json!({"path":"/tmp/image.png"})).is_none());
        assert!(immutable_key(&json!({"url":"data:image/png;base64,AA=="})).is_some());
        let a=immutable_key(&json!({"source":{"data":"AA==","media_type":"image/png"}}));
        let b=immutable_key(&json!({"source":{"data":"AQ==","media_type":"image/png"}}));
        assert!(a.is_some());assert_ne!(a,b);
        assert_ne!(immutable_key(&json!({"data":"AA==","detail":"low"})),immutable_key(&json!({"data":"AA==","detail":"high"})));
        let huge=json!({"data":"A".repeat(2<<20)});
        assert_eq!(immutable_key(&huge).unwrap().len(),64,"cache must not retain full base64 keys");
    }
    #[test]
    fn feature_lru_keeps_inflight_arcs_and_clears_generation() {
        let mut c=FeatureCache::default();c.limit=400;c.bind("encoder-a");
        c.put("a".into(),json!({}),FeatureFile::create(&[1u8;64]).unwrap());
        let (_,held)=c.get("a").unwrap();
        c.put("b".into(),json!({}),FeatureFile::create(&[2u8;64]).unwrap());
        assert!(c.get("a").is_none());assert_eq!(std::fs::read(held.path()).unwrap()[0],1);assert!(c.bytes<=c.limit);
        assert!(c.get("b").is_some());c.bind("encoder-b");
        assert!(c.get("b").is_none());assert_eq!(c.bytes,0);
    }
    #[test]
    #[ignore = "requires DSGB10_TEST_TOKENIZER"]
    fn appended_image_after_noncanonical_generation_uses_canonical_positions() {
        use crate::session::SessionTable;
        let tok=Tokenizer::from_file(&std::env::var("DSGB10_TEST_TOKENIZER").unwrap()).unwrap();
        for mode in ["scope-a", "scope-b"] {
            let scope=format!("canonical:{mode}");
            let mut table=SessionTable::new(8,1<<20);
            let raw=format!("<｜begin▁of▁sentence｜><｜System｜>Preserve this deliberately long shared prefix and inspect every actual picture in order.<｜User｜>{IMAGE_PLACEHOLDER}<｜Assistant｜></think>");
            let meta=vec![json!({"rows":32,"dim":DIM,"digest":"a1".repeat(32)})];
            let a=expand(&raw,&scope,&tok,1,&"11".repeat(32),&meta,vec![vec![0;32*DIM*2]]).unwrap();
            let (first,_)=table.plan_images(&a.prompt,&a.scope,&a.cache_images,&tok,1);
            let mut gen:Vec<u32>="Hi there".chars().map(|c|tok.encode(&c.to_string())[0]).collect();
            gen.push(tok.token_id("<｜end▁of▁sentence｜>").unwrap());
            table.finish(first.session_id,&a.prompt,&first.ids,&gen,&tok);
            let raw2=format!("{raw}Hi there<｜end▁of▁sentence｜><｜User｜>{IMAGE_PLACEHOLDER}Next image?<｜Assistant｜></think>");
            let mut meta2=meta.clone();meta2.push(json!({"rows":32,"dim":DIM,"digest":"b2".repeat(32)}));
            let b=expand(&raw2,&scope,&tok,1,&"11".repeat(32),&meta2,vec![vec![0;32*DIM*2];2]).unwrap();
            let (next,_)=table.plan_images(&b.prompt,&b.scope,&b.cache_images,&tok,1);
            assert_eq!(next.ids,tok.encode(&b.prompt));
            assert!(next.reused>=first.ids.len());
            assert!(next.reused<first.ids.len()+gen.len());
            for im in &b.images {
                assert!(next.ids[im.start as usize..(im.start+im.rows) as usize].iter().all(|&x|x==IMAGE_ID));
            }
        }
    }
    #[test]
    #[ignore = "requires DSGB10_TEST_TOKENIZER"]
    fn incremental_images_clip_only_changed_prefix_and_persist() {
        use crate::session::SessionTable;
        let tok=Tokenizer::from_file(&std::env::var("DSGB10_TEST_TOKENIZER").unwrap()).unwrap();
        for mode in ["scope-a", "scope-b"] {
            let mut table=SessionTable::new(32,1<<20);
            let scope=format!("1017:asset:{mode}");
            let mut raw="<｜begin▁of▁sentence｜><｜System｜>Verify actual image pixels and preserve exactly the ordered historical images. This stable instruction has more than sixty four bytes.".to_string();
            let mut meta=Vec::new();
            let mut previous=0;
            for n in 1..=12 {
                raw.push_str(&format!("<｜User｜>Screenshot {n}: {IMAGE_PLACEHOLDER}<｜Assistant｜>Received.<｜end▁of▁sentence｜>"));
                meta.push(json!({"rows":32,"dim":DIM,"digest":format!("{n:064x}")}));
                let v=expand(&raw,&scope,&tok,1,&"11".repeat(32),&meta,vec![vec![0;32*DIM*2];n]).unwrap();
                let (p,_)=table.plan_images(&v.prompt,&v.scope,&v.cache_images,&tok,1);
                if n>1 { assert!(p.reused>=previous, "append lost old prefix at {n}"); }
                previous=p.ids.len();
                table.finish(p.session_id,&v.prompt,&p.ids,&[],&tok);
                if n==12 {
                    let mut changed=meta.clone();changed[5]["digest"]=json!("fe".repeat(32));
                    let x=expand(&raw,&scope,&tok,1,&"11".repeat(32),&changed,vec![vec![0;32*DIM*2];n]).unwrap();
                    let (q,_)=table.plan_images(&x.prompt,&x.scope,&x.cache_images,&tok,1);
                    assert!(q.reused>v.images[4].start as usize);
                    assert!(q.reused<=v.images[5].start as usize);
                    assert_ne!(q.session_id,p.session_id);
                    table.finish(q.session_id,&x.prompt,&q.ids,&[],&tok);
                    let dir=std::env::temp_dir().join(format!("dsfront-prefix-{}-{mode}",std::process::id()));
                    std::fs::create_dir_all(&dir).unwrap();let path=dir.join("sessions");
                    table.save(&path,3600).unwrap();
                    let mut fresh=SessionTable::new(32,1<<20);fresh.load(&path,&tok,3600).unwrap();
                    let (q,_)=fresh.plan_images(&x.prompt,&x.scope,&x.cache_images,&tok,1);
                    assert!(q.reused>=previous-2);
                    let (isolated,_)=fresh.plan_images(&x.prompt,&format!("{scope}-OTHER"),&x.cache_images,&tok,1);
                    assert_eq!(isolated.reused,0);
                    std::fs::remove_dir(&dir).unwrap();
                }
            }
        }
    }

    #[test]
    fn strict_digest() {
        assert_eq!(digest(&"a1".repeat(32)).unwrap(), [0xa1;32]);
        assert!(digest(&"x".repeat(64)).is_err());
        assert!(digest(&"a".repeat(63)).is_err());
    }

    #[test]
    #[ignore = "requires DSGB10_TEST_TOKENIZER"]
    fn accumulated_image_history_preserves_order_and_identity() {
        let tok=Tokenizer::from_file(&std::env::var("DSGB10_TEST_TOKENIZER").unwrap()).unwrap();
        let raw=(0..64).map(|i| format!("Turn {i}: {IMAGE_PLACEHOLDER}\n")).collect::<String>();
        let meta=(0..64).map(|i|json!({"rows":2,"dim":DIM,"digest":format!("{i:064x}")})).collect::<Vec<_>>();
        let blobs=vec![vec![0;2*DIM*2];64];
        let result=expand(&raw,"history:scope-a",&tok,1,&"11".repeat(32),&meta,blobs.clone()).unwrap();
        let ids=tok.encode(&result.prompt);
        assert_eq!(result.images.len(),64);
        assert_eq!(ids.iter().filter(|&&id|id==IMAGE_ID).count(),128);
        for span in &result.images {
            assert!(ids[span.start as usize..span.start as usize+span.rows as usize].iter().all(|&id|id==IMAGE_ID));
        }
        let mut changed=meta.clone();changed.swap(0,63);
        assert_ne!(result.cache_images,expand(&raw,"history:scope-a",&tok,1,&"11".repeat(32),&changed,blobs.clone()).unwrap().cache_images);
        assert_ne!(result.scope,expand(&raw,"history:scope-b",&tok,1,&"11".repeat(32),&meta,blobs).unwrap().scope);
    }

    #[test]
    #[ignore = "requires DSGB10_TEST_TOKENIZER; deployment runs with --include-ignored"]
    fn expansion_and_cache_isolation() {
        use crate::session::SessionTable;
        let tok = Tokenizer::from_file(&std::env::var("DSGB10_TEST_TOKENIZER").unwrap()).unwrap();
        let raw = format!("<｜begin▁of▁sentence｜><｜System｜>Carefully inspect the following image. Explain only what the actual pixels show. This shared instruction is sufficiently long to exercise session reuse.<｜User｜>{IMAGE_PLACEHOLDER}What shape?<｜Assistant｜></think>");
        let identity = "1a".repeat(32);
        let meta = vec![json!({"rows":32,"dim":5120,"digest":"a2".repeat(32)})];
        let blobs = || vec![vec![0u8;32*DIM*2]];
        let a = expand(&raw,"revision-A:scope-a",&tok,1,&identity,&meta,blobs()).unwrap();
        let ids=tok.encode(&a.prompt);
        assert_eq!(ids.iter().filter(|&&x| x==IMAGE_ID).count(),32);
        let image=&a.images[0];
        assert!(ids[image.start as usize..image.start as usize+32].iter().all(|&x|x==IMAGE_ID));
        let mut sessions=SessionTable::new(32,1<<20);
        let (first,_)=sessions.plan_images(&a.prompt,&a.scope,&a.cache_images,&tok,1);
        sessions.finish(first.session_id,&a.prompt,&first.ids,&tok.encode("A circle."),&tok);
        let (warm,_)=sessions.plan_images(&a.prompt,&a.scope,&a.cache_images,&tok,1);
        assert_eq!(warm.session_id,first.session_id);assert!(warm.reused>0);
        sessions.finish(warm.session_id,&a.prompt,&warm.ids,&tok.encode("A circle."),&tok);
        let variants=[
            expand(&raw,"revision-A:scope-b",&tok,1,&identity,&meta,blobs()).unwrap(),
            expand(&raw,"revision-B:scope-a",&tok,1,&identity,&meta,blobs()).unwrap(),
            expand(&raw,"revision-A:scope-a",&tok,1,&"1b".repeat(32),&meta,blobs()).unwrap(),
            expand(&raw,"revision-A:scope-a",&tok,1,&identity,&[json!({"rows":32,"dim":5120,"digest":"a3".repeat(32)})],blobs()).unwrap(),
            expand(&raw.replace("What shape?","What color?"),"revision-A:scope-a",&tok,1,&identity,&meta,blobs()).unwrap(),
        ];
        for (i,v) in variants.iter().enumerate() {
            let (p,_)=sessions.plan_images(&v.prompt,&v.scope,&v.cache_images,&tok,1);
            if i<3 {assert_ne!(p.session_id,first.session_id);assert_eq!(p.base_session_id,0);assert_eq!(p.reused,0);}
            else {assert!(p.reused>0); if i==3 { assert!(p.reused <= a.images[0].start as usize); }}
            sessions.finish(p.session_id,&v.prompt,&p.ids,&[],&tok);
        }
        let changed_position=expand(&raw.replace("<｜User｜>","<｜User｜>Before the picture: "),"revision-A:scope-a",&tok,1,&identity,&meta,blobs()).unwrap();
        assert_ne!(a.cache_images,changed_position.cache_images);
        let dir=std::env::temp_dir().join(format!("dsfront-vision-scopes-{}",std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();let path=dir.join("sessions");
        sessions.save(&path,3600).unwrap();
        let mut reloaded=SessionTable::new(32,1<<20);reloaded.load(&path,&tok,3600).unwrap();
        let (p,_)=reloaded.plan_images(&a.prompt,&a.scope,&a.cache_images,&tok,1);assert!(p.reused>0);
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    #[ignore = "requires DSGB10_TEST_TOKENIZER"]
    fn reject_bad_spans_and_literal_image_tokens() {
        let tok = Tokenizer::from_file(&std::env::var("DSGB10_TEST_TOKENIZER").unwrap()).unwrap();
        let meta=vec![json!({"rows":1,"dim":5120,"digest":"a1".repeat(32)})];
        let good=vec![vec![0u8;DIM*2]];
        let mut bad=good.clone();bad[0][0]=0x80;bad[0][1]=0x7f;
        assert!(expand(IMAGE_PLACEHOLDER,"normal",&tok,1,&"00".repeat(32),&meta,bad).is_err());
        assert!(expand("No image here","normal",&tok,1,&"00".repeat(32),&meta,good.clone()).is_err());
        assert!(expand(IMAGE_PLACEHOLDER,"normal",&tok,1,&"00".repeat(32),&meta,vec![vec![0;2]]).is_err());
    }
}
