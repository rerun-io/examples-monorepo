//! A minimal ISO-BMFF (MP4) demuxer for H.264 and HEVC video tracks.
//!
//! It reads only what a remux needs: the top-level box walk (32-bit, 64-bit and to-end-of-file sizes), `moov`
//! wherever it sits, and for each `avc1`/`avc3` or `hvc1`/`hev1` track the parameter sets of its `avcC` or `hvcC` and
//! the sample table. Box layouts follow ISO/IEC 14496-12 (sample tables, edit lists) and ISO/IEC 14496-15 sections
//! 5.3.3 (`avcC`) and 8.3.3 (`hvcC`).
use std::io::{Read, Seek, SeekFrom};

/// Why a file could not be demuxed.
#[derive(Debug, thiserror::Error)]
pub enum Mp4Error {
    /// The underlying reader failed.
    #[error("mp4 read failed: {0}")]
    Io(#[from] std::io::Error),
    /// A box or table is shorter than its header says, or a field is out of range.
    #[error("malformed mp4: {0}")]
    Malformed(String),
    /// The file has no `moov` box.
    #[error("mp4 has no moov box")]
    NoMoov,
}

/// One coded sample of a track.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    /// Absolute byte offset of the sample in the file.
    pub offset: u64,
    /// Sample size in bytes.
    pub size: u32,
    /// Decode time in track timescale units.
    pub dts: i64,
    /// Presentation time in track timescale units: DTS plus the `ctts` offset, minus the edit list's media time.
    pub pts: i64,
    /// Whether the sample is a sync sample (`stss`; every sample when the table is absent).
    pub is_sync: bool,
}

/// The video codec of a track.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 / AVC (`avc1`, `avc3`).
    H264,
    /// H.265 / HEVC (`hvc1`, `hev1`).
    H265,
}

/// One H.264 or HEVC video track: its decoder configuration and its samples in decode order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoTrack {
    /// `tkhd` track id.
    pub track_id: u32,
    /// The codec of the sample entry.
    pub codec: VideoCodec,
    /// Coded width from the sample entry, pixels.
    pub width: u16,
    /// Coded height from the sample entry, pixels.
    pub height: u16,
    /// `mdhd` timescale, ticks per second.
    pub timescale: u32,
    /// Bytes in each NAL length prefix (`lengthSizeMinusOne + 1`).
    pub nal_length_size: u8,
    /// The parameter sets without length prefixes: SPS and PPS from `avcC`, or VPS, SPS and PPS from `hvcC`, in that
    /// order.
    pub parameter_sets: Vec<Vec<u8>>,
    /// Samples in decode order.
    pub samples: Vec<Sample>,
}

fn malformed(what: impl Into<String>) -> Mp4Error {
    Mp4Error::Malformed(what.into())
}

/// The `N` bytes at `at`, or an error naming the field past the end of the box.
fn field<const N: usize>(b: &[u8], at: usize) -> Result<[u8; N], Mp4Error> {
    b.get(at..)
        .and_then(|rest| rest.first_chunk::<N>())
        .copied()
        .ok_or_else(|| {
            malformed(format!(
                "field at {at}+{N} past the end of a {}-byte box",
                b.len()
            ))
        })
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, Mp4Error> {
    Ok(u32::from_be_bytes(field(b, at)?))
}

/// Child boxes of a box payload: `(type, payload)` pairs. Sizes 0 (to the end) and 1 (64-bit) are supported.
/// A child box: its type and its payload.
type Child<'a> = ([u8; 4], &'a [u8]);

fn children(b: &[u8]) -> Result<Vec<Child<'_>>, Mp4Error> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 8 <= b.len() {
        let size = u32_at(b, at)? as u64;
        let kind: [u8; 4] = [b[at + 4], b[at + 5], b[at + 6], b[at + 7]];
        let (header, size) = match size {
            0 => (8, (b.len() - at) as u64),
            1 => (16, u64::from_be_bytes(field(b, at + 8)?)),
            s => (8, s),
        };
        let end = usize::try_from(size)
            .ok()
            .and_then(|s| at.checked_add(s))
            .filter(|&e| e <= b.len() && size >= header as u64);
        let end = end.ok_or_else(|| {
            malformed(format!(
                "box {} at {at} has size {size}",
                String::from_utf8_lossy(&kind)
            ))
        })?;
        out.push((kind, &b[at + header..end]));
        at = end;
    }
    Ok(out)
}

fn child<'a>(b: &'a [u8], kind: &[u8; 4]) -> Result<Option<&'a [u8]>, Mp4Error> {
    Ok(children(b)?
        .into_iter()
        .find(|(k, _)| k == kind)
        .map(|(_, p)| p))
}

fn need<'a>(b: &'a [u8], kind: &[u8; 4]) -> Result<&'a [u8], Mp4Error> {
    child(b, kind)?
        .ok_or_else(|| malformed(format!("missing {} box", String::from_utf8_lossy(kind))))
}

/// Find `moov` among the top-level boxes and return its payload. The walk seeks over every other box, so a `moov`
/// at the end of a multi-gigabyte file costs one read of its own size.
pub fn read_moov<R: Read + Seek>(reader: &mut R) -> Result<Vec<u8>, Mp4Error> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    let mut at = 0u64;
    while at + 8 <= file_len {
        reader.seek(SeekFrom::Start(at))?;
        let mut header = [0u8; 16];
        reader.read_exact(&mut header[..8])?;
        let size32 = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        let (header_len, size) = match size32 {
            0 => (8u64, file_len - at),
            1 => {
                reader.read_exact(&mut header[8..16])?;
                (16, u64::from_be_bytes(field(&header, 8)?))
            }
            s => (8, u64::from(s)),
        };
        if size < header_len || at.checked_add(size).is_none_or(|end| end > file_len) {
            return Err(malformed(format!(
                "top-level box at {at} has size {size} in a {file_len}-byte file"
            )));
        }
        if &header[4..8] == b"moov" {
            let len =
                usize::try_from(size - header_len).map_err(|_| malformed("moov too large"))?;
            let mut moov = vec![0u8; len];
            reader.read_exact(&mut moov)?;
            return Ok(moov);
        }
        at += size;
    }
    Err(Mp4Error::NoMoov)
}

/// Read every H.264 and HEVC video track of a file. Tracks of other codecs and handlers are skipped.
pub fn read_video_tracks<R: Read + Seek>(reader: &mut R) -> Result<Vec<VideoTrack>, Mp4Error> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    let moov = read_moov(reader)?;
    let mut tracks = Vec::new();
    for (kind, trak) in children(&moov)? {
        if &kind == b"trak" {
            tracks.extend(parse_trak(trak, file_len)?);
        }
    }
    Ok(tracks)
}

fn full_box_version(b: &[u8]) -> Result<u8, Mp4Error> {
    Ok(field::<1>(b, 0)?[0])
}

fn parse_trak(trak: &[u8], file_len: u64) -> Result<Option<VideoTrack>, Mp4Error> {
    let tkhd = need(trak, b"tkhd")?;
    let track_id = u32_at(tkhd, if full_box_version(tkhd)? == 1 { 20 } else { 12 })?;
    let mdia = need(trak, b"mdia")?;
    if &need(mdia, b"hdlr")?
        .get(8..12)
        .ok_or_else(|| malformed("short hdlr"))?
        != b"vide"
    {
        return Ok(None);
    }
    let mdhd = need(mdia, b"mdhd")?;
    let timescale = u32_at(mdhd, if full_box_version(mdhd)? == 1 { 20 } else { 12 })?;
    let stbl = need(need(mdia, b"minf")?, b"stbl")?;
    let stsd = need(stbl, b"stsd")?;
    let Some((fourcc, entry)) = children(stsd.get(8..).ok_or_else(|| malformed("short stsd"))?)?
        .into_iter()
        .next()
    else {
        return Err(malformed("empty stsd"));
    };
    let (codec, config): (VideoCodec, &[u8; 4]) = match &fourcc {
        b"avc1" | b"avc3" => (VideoCodec::H264, b"avcC"),
        b"hvc1" | b"hev1" => (VideoCodec::H265, b"hvcC"),
        _ => return Ok(None),
    };
    // VisualSampleEntry: 6 reserved, data_reference_index, 16 pre-defined/reserved, width, height, then 50 more bytes.
    let (width, height) = (
        u16::from_be_bytes(field(entry, 24)?),
        u16::from_be_bytes(field(entry, 26)?),
    );
    let config = need(
        entry
            .get(78..)
            .ok_or_else(|| malformed("short visual sample entry"))?,
        config,
    )?;
    let (nal_length_size, parameter_sets) = match codec {
        VideoCodec::H264 => parse_avcc(config)?,
        VideoCodec::H265 => parse_hvcc(config)?,
    };
    let media_time = match child(trak, b"edts")?
        .map(|edts| need(edts, b"elst"))
        .transpose()?
    {
        Some(elst) => first_media_time(elst)?,
        None => 0,
    };
    let samples = sample_table(stbl, media_time, file_len)?;
    Ok(Some(VideoTrack {
        track_id,
        codec,
        width,
        height,
        timescale,
        nal_length_size,
        parameter_sets,
        samples,
    }))
}

/// `avcC`: the NAL length size and the SPS and PPS arrays (ISO/IEC 14496-15 section 5.3.3.1.2).
fn parse_avcc(b: &[u8]) -> Result<(u8, Vec<Vec<u8>>), Mp4Error> {
    let nal_length_size = (field::<1>(b, 4)?[0] & 0x03) + 1;
    let mut at = 5usize;
    let mut sets = Vec::new();
    // The SPS count sits in the low five bits of its byte; the PPS count fills its byte.
    for mask in [0x1f, 0xff] {
        let count = field::<1>(b, at)?[0] & mask;
        at += 1;
        for _ in 0..count {
            let len = usize::from(u16::from_be_bytes(field(b, at)?));
            let nal = b
                .get(at + 2..at + 2 + len)
                .ok_or_else(|| malformed("avcC NAL past the end"))?;
            sets.push(nal.to_vec());
            at += 2 + len;
        }
    }
    Ok((nal_length_size, sets))
}

/// `hvcC`: the NAL length size and the VPS, SPS and PPS arrays (ISO/IEC 14496-15 section 8.3.3.1.2).
fn parse_hvcc(b: &[u8]) -> Result<(u8, Vec<Vec<u8>>), Mp4Error> {
    let nal_length_size = (field::<1>(b, 21)?[0] & 0x03) + 1;
    let arrays = usize::from(field::<1>(b, 22)?[0]);
    let mut at = 23usize;
    let mut sets: Vec<(u8, Vec<u8>)> = Vec::new();
    for _ in 0..arrays {
        let nal_type = field::<1>(b, at)?[0] & 0x3f;
        let count = usize::from(u16::from_be_bytes(field(b, at + 1)?));
        at += 3;
        for _ in 0..count {
            let len = usize::from(u16::from_be_bytes(field(b, at)?));
            let nal = b
                .get(at + 2..at + 2 + len)
                .ok_or_else(|| malformed("hvcC NAL past the end"))?;
            if (32..=34).contains(&nal_type) {
                sets.push((nal_type, nal.to_vec()));
            }
            at += 2 + len;
        }
    }
    sets.sort_by_key(|(t, _)| *t);
    Ok((nal_length_size, sets.into_iter().map(|(_, n)| n).collect()))
}

/// The media time of the first non-empty edit (`media_time == -1` is an empty edit).
fn first_media_time(elst: &[u8]) -> Result<i64, Mp4Error> {
    let v1 = full_box_version(elst)? == 1;
    let (entry, count) = (if v1 { 20 } else { 12 }, u32_at(elst, 4)? as usize);
    for i in 0..count {
        let at = 8 + i * entry;
        let media_time = if v1 {
            i64::from_be_bytes(field(elst, at + 8)?)
        } else {
            i64::from(u32_at(elst, at + 4)? as i32)
        };
        if media_time != -1 {
            return Ok(media_time);
        }
    }
    Ok(0)
}

/// The samples of a track. `file_len` bounds a fixed-size `stsz`: the samples' bytes lie in the file.
fn sample_table(stbl: &[u8], media_time: i64, file_len: u64) -> Result<Vec<Sample>, Mp4Error> {
    let stsz = need(stbl, b"stsz")?;
    let (fixed, count) = (u32_at(stsz, 4)?, u32_at(stsz, 8)? as usize);
    // Bound the sample count before anything is allocated from it: by the stsz box size (one entry per sample), or
    // for a fixed sample size by the file size, and then by the stts entry counts, which must add up to it.
    let in_bounds = if fixed == 0 {
        count
            .checked_mul(4)
            .and_then(|n| n.checked_add(12))
            .is_some_and(|end| end <= stsz.len())
    } else {
        (count as u64)
            .checked_mul(u64::from(fixed))
            .is_some_and(|bytes| bytes <= file_len)
    };
    if !in_bounds {
        return Err(malformed(format!(
            "stsz states {count} samples of size {fixed} in a {}-byte box of a {file_len}-byte file",
            stsz.len()
        )));
    }
    let stts = need(stbl, b"stts")?;
    let mut stts_total = 0usize;
    for i in 0..u32_at(stts, 4)? as usize {
        stts_total += u32_at(stts, 8 + 8 * i)? as usize;
        if stts_total > count {
            break;
        }
    }
    if stts_total != count {
        return Err(malformed(format!(
            "stts covers {stts_total} or more samples, stsz has {count}"
        )));
    }
    let sizes: Vec<u32> = (0..count)
        .map(|i| {
            if fixed != 0 {
                Ok(fixed)
            } else {
                u32_at(stsz, 12 + 4 * i)
            }
        })
        .collect::<Result<_, _>>()?;

    let chunk_offsets: Vec<u64> = if let Some(stco) = child(stbl, b"stco")? {
        (0..u32_at(stco, 4)? as usize)
            .map(|i| Ok(u64::from(u32_at(stco, 8 + 4 * i)?)))
            .collect::<Result<_, Mp4Error>>()?
    } else {
        let co64 = need(stbl, b"co64")?;
        (0..u32_at(co64, 4)? as usize)
            .map(|i| Ok(u64::from_be_bytes(field(co64, 8 + 8 * i)?)))
            .collect::<Result<_, Mp4Error>>()?
    };

    // stsc runs: chunks [first_chunk, next.first_chunk) hold samples_per_chunk samples each.
    let stsc = need(stbl, b"stsc")?;
    let runs: Vec<(usize, usize)> = (0..u32_at(stsc, 4)? as usize)
        .map(|i| {
            Ok((
                u32_at(stsc, 8 + 12 * i)? as usize,
                u32_at(stsc, 12 + 12 * i)? as usize,
            ))
        })
        .collect::<Result<_, Mp4Error>>()?;
    let mut offsets = Vec::with_capacity(count);
    for (r, &(first, per_chunk)) in runs.iter().enumerate() {
        let last = runs.get(r + 1).map_or(Ok(chunk_offsets.len()), |next| {
            next.0
                .checked_sub(1)
                .ok_or_else(|| malformed("stsc run starts at chunk 0"))
        })?;
        for chunk in first..=last {
            let mut offset = *chunk_offsets
                .get(chunk.wrapping_sub(1))
                .ok_or_else(|| malformed(format!("stsc names chunk {chunk}")))?;
            for _ in 0..per_chunk {
                let size = *sizes
                    .get(offsets.len())
                    .ok_or_else(|| malformed("stsc describes more samples than stsz"))?;
                offsets.push(offset);
                offset = offset
                    .checked_add(u64::from(size))
                    .ok_or_else(|| malformed("sample offset overflows 64 bits"))?;
            }
        }
    }
    if offsets.len() != count {
        return Err(malformed(format!(
            "stsc covers {} samples, stsz has {count}",
            offsets.len()
        )));
    }

    let mut dts = Vec::with_capacity(count);
    let mut t = 0i64;
    for i in 0..u32_at(stts, 4)? as usize {
        let (n, delta) = (
            u32_at(stts, 8 + 8 * i)?,
            i64::from(u32_at(stts, 12 + 8 * i)?),
        );
        for _ in 0..n {
            dts.push(t);
            t = t
                .checked_add(delta)
                .ok_or_else(|| malformed("stts decode time overflows 64 bits"))?;
        }
    }
    let mut composition = vec![0i64; count];
    if let Some(ctts) = child(stbl, b"ctts")? {
        let mut s = 0usize;
        for i in 0..u32_at(ctts, 4)? as usize {
            let (n, offset) = (
                u32_at(ctts, 8 + 8 * i)? as usize,
                i64::from(u32_at(ctts, 12 + 8 * i)? as i32),
            );
            for slot in composition.iter_mut().skip(s).take(n) {
                *slot = offset;
            }
            s += n;
        }
    }
    let sync: Option<Vec<u32>> = child(stbl, b"stss")?
        .map(|stss| {
            (0..u32_at(stss, 4)? as usize)
                .map(|i| u32_at(stss, 8 + 4 * i))
                .collect::<Result<_, _>>()
        })
        .transpose()?;
    if dts.len() != count {
        return Err(malformed(format!(
            "stts covers {} samples, stsz has {count}",
            dts.len()
        )));
    }
    (0..count)
        .map(|i| {
            Ok(Sample {
                offset: offsets[i],
                size: sizes[i],
                dts: dts[i],
                pts: dts[i]
                    .checked_add(composition[i])
                    .and_then(|t| t.checked_sub(media_time))
                    .ok_or_else(|| malformed("presentation time overflows 64 bits"))?,
                is_sync: sync
                    .as_ref()
                    .is_none_or(|s| s.binary_search(&(i as u32 + 1)).is_ok()),
            })
        })
        .collect()
}

/// Builders for synthetic MP4 files, shared by this crate's tests.
#[cfg(test)]
pub(crate) mod testing {
    /// A box: big-endian 32-bit size, type, payload.
    pub fn bx(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    /// A full box: version and flags, then the payload.
    pub fn full(kind: &[u8; 4], version: u8, payload: &[u8]) -> Vec<u8> {
        let mut body = vec![version, 0, 0, 0];
        body.extend_from_slice(payload);
        bx(kind, &body)
    }

    fn table(rows: &[&[u32]]) -> Vec<u8> {
        let mut out = (rows.len() as u32).to_be_bytes().to_vec();
        for row in rows {
            for v in *row {
                out.extend_from_slice(&v.to_be_bytes());
            }
        }
        out
    }

    /// The parts of one synthetic HEVC (or, with `h264`, H.264) track.
    pub struct TrackSpec<'a> {
        pub h264: bool,
        pub track_id: u32,
        pub sizes: &'a [u32],
        pub chunk_offsets: &'a [u64],
        pub stsc: &'a [[u32; 3]],
        pub stts: &'a [[u32; 2]],
        pub ctts: &'a [[u32; 2]],
        pub stss: Option<&'a [u32]>,
        pub media_time: i32,
        pub co64: bool,
        /// `(nal type, nal)`: HEVC types 32-34, or H.264 types 7 (SPS) and 8 (PPS).
        pub parameter_sets: &'a [(u8, &'a [u8])],
    }

    /// `avcC` with 4-byte NAL lengths: the type-7 sets, then the type-8 sets.
    fn avcc(parameter_sets: &[(u8, &[u8])]) -> Vec<u8> {
        let mut out = vec![1, 0x64, 0, 0x28, 0xfc | 3];
        for (t, mask) in [(7u8, 0xe0u8), (8, 0)] {
            let sets: Vec<&[u8]> = parameter_sets
                .iter()
                .filter(|(k, _)| *k == t)
                .map(|(_, n)| *n)
                .collect();
            out.push(mask | sets.len() as u8);
            for nal in sets {
                out.extend_from_slice(&(nal.len() as u16).to_be_bytes());
                out.extend_from_slice(nal);
            }
        }
        out
    }

    /// A `trak` box for an `hvc1` (or `avc1`) track with the given tables.
    pub fn trak(spec: &TrackSpec) -> Vec<u8> {
        let mut tkhd = vec![0u8; 8];
        tkhd.extend_from_slice(&spec.track_id.to_be_bytes());
        tkhd.extend_from_slice(&[0u8; 68]);
        let mut mdhd = vec![0u8; 8];
        mdhd.extend_from_slice(&30000u32.to_be_bytes());
        mdhd.extend_from_slice(&[0u8; 8]);
        let mut hdlr = vec![0u8; 4];
        hdlr.extend_from_slice(b"vide");
        hdlr.extend_from_slice(&[0u8; 13]);
        let mut hvcc = vec![1u8; 21];
        hvcc.push(0xfc | 3);
        hvcc.push(spec.parameter_sets.len() as u8);
        for (t, nal) in spec.parameter_sets {
            hvcc.push(0x80 | t);
            hvcc.extend_from_slice(&1u16.to_be_bytes());
            hvcc.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            hvcc.extend_from_slice(nal);
        }
        let mut entry = vec![0u8; 78];
        entry[24..26].copy_from_slice(&1920u16.to_be_bytes());
        entry[26..28].copy_from_slice(&1080u16.to_be_bytes());
        if spec.h264 {
            entry.extend(bx(b"avcC", &avcc(spec.parameter_sets)));
        } else {
            entry.extend(bx(b"hvcC", &hvcc));
        }
        let mut stsd_body = 1u32.to_be_bytes().to_vec();
        stsd_body.extend(bx(if spec.h264 { b"avc1" } else { b"hvc1" }, &entry));
        let mut stsz = vec![0u8; 4];
        stsz.extend_from_slice(&(spec.sizes.len() as u32).to_be_bytes());
        for s in spec.sizes {
            stsz.extend_from_slice(&s.to_be_bytes());
        }
        let offsets = if spec.co64 {
            let mut b = (spec.chunk_offsets.len() as u32).to_be_bytes().to_vec();
            for o in spec.chunk_offsets {
                b.extend_from_slice(&o.to_be_bytes());
            }
            full(b"co64", 0, &b)
        } else {
            let rows: Vec<[u32; 1]> = spec.chunk_offsets.iter().map(|&o| [o as u32]).collect();
            full(
                b"stco",
                0,
                &table(&rows.iter().map(|r| &r[..]).collect::<Vec<_>>()),
            )
        };
        let mut stbl = full(b"stsd", 0, &stsd_body);
        stbl.extend(full(
            b"stts",
            0,
            &table(&spec.stts.iter().map(|r| &r[..]).collect::<Vec<_>>()),
        ));
        stbl.extend(full(
            b"ctts",
            0,
            &table(&spec.ctts.iter().map(|r| &r[..]).collect::<Vec<_>>()),
        ));
        stbl.extend(full(
            b"stsc",
            0,
            &table(&spec.stsc.iter().map(|r| &r[..]).collect::<Vec<_>>()),
        ));
        stbl.extend(full(b"stsz", 0, &stsz));
        stbl.extend(offsets);
        if let Some(stss) = spec.stss {
            let rows: Vec<[u32; 1]> = stss.iter().map(|&s| [s]).collect();
            stbl.extend(full(
                b"stss",
                0,
                &table(&rows.iter().map(|r| &r[..]).collect::<Vec<_>>()),
            ));
        }
        let mut elst = 1u32.to_be_bytes().to_vec();
        elst.extend_from_slice(&0u32.to_be_bytes());
        elst.extend_from_slice(&spec.media_time.to_be_bytes());
        elst.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        let mut mdia = full(b"mdhd", 0, &mdhd);
        mdia.extend(full(b"hdlr", 0, &hdlr));
        mdia.extend(bx(b"minf", &bx(b"stbl", &stbl)));
        let mut trak = full(b"tkhd", 0, &tkhd);
        trak.extend(bx(b"edts", &full(b"elst", 0, &elst)));
        trak.extend(bx(b"mdia", &mdia));
        bx(b"trak", &trak)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use std::io::Cursor;

    const VPS: &[u8] = &[0x40, 0x01, 0xaa];
    const SPS: &[u8] = &[0x42, 0x01, 0xbb];
    const PPS: &[u8] = &[0x44, 0x01, 0xcc];

    fn spec<'a>(
        sizes: &'a [u32],
        offsets: &'a [u64],
        stsc: &'a [[u32; 3]],
        co64: bool,
    ) -> TrackSpec<'a> {
        TrackSpec {
            h264: false,
            track_id: 1,
            sizes,
            chunk_offsets: offsets,
            stsc,
            stts: &[[4, 1001]],
            ctts: &[[2, 0], [2, 2002]],
            stss: Some(&[1, 3]),
            media_time: 1001,
            co64,
            parameter_sets: &[(34, PPS), (32, VPS), (33, SPS)],
        }
    }

    #[test]
    fn an_avc1_track_reads_its_sps_and_pps_from_avcc() {
        let (sps, pps): (&[u8], &[u8]) = (&[0x67, 0x64, 0x00, 0x28], &[0x68, 0xee, 0x3c]);
        let spec = TrackSpec {
            h264: true,
            parameter_sets: &[(8, pps), (7, sps)],
            ..spec(&[10, 20, 30, 40], &[100, 200], &[[1, 2, 1]], false)
        };
        let tracks = read_video_tracks(&mut Cursor::new(bx(b"moov", &trak(&spec)))).unwrap();
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].codec, VideoCodec::H264);
        assert_eq!(tracks[0].parameter_sets, vec![sps.to_vec(), pps.to_vec()]);
        assert_eq!(tracks[0].nal_length_size, 4);
        assert_eq!(tracks[0].samples.len(), 4);
        // An avcC whose SPS runs past its end is refused, not read past it.
        assert!(parse_avcc(&[1, 0x64, 0, 0x28, 0xff, 0xe1, 0, 9, 0x67]).is_err());
    }

    #[test]
    fn moov_at_the_end_after_a_64_bit_mdat_is_found() {
        let mut file = bx(b"ftyp", b"isom");
        // A 64-bit mdat header (size == 1) with 4 payload bytes.
        file.extend_from_slice(&1u32.to_be_bytes());
        file.extend_from_slice(b"mdat");
        file.extend_from_slice(&20u64.to_be_bytes());
        file.extend_from_slice(&[9, 9, 9, 9]);
        file.extend(bx(
            b"moov",
            &trak(&spec(&[10, 20, 30, 40], &[100, 200], &[[1, 2, 1]], false)),
        ));
        file.extend(bx(b"inst", &[7u8; 12]));
        let tracks = read_video_tracks(&mut Cursor::new(file)).unwrap();
        assert_eq!(tracks.len(), 1);
        assert_eq!(
            tracks[0].parameter_sets,
            vec![VPS.to_vec(), SPS.to_vec(), PPS.to_vec()]
        );
        assert_eq!(tracks[0].nal_length_size, 4);
        assert_eq!(
            (tracks[0].width, tracks[0].height, tracks[0].timescale),
            (1920, 1080, 30000)
        );
    }

    #[test]
    fn a_size_zero_box_runs_to_the_end_of_the_file() {
        let mut file = bx(b"ftyp", b"isom");
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(b"moov");
        file.extend(trak(&spec(
            &[10, 20, 30, 40],
            &[100, 200],
            &[[1, 2, 1]],
            false,
        )));
        assert_eq!(read_video_tracks(&mut Cursor::new(file)).unwrap().len(), 1);
    }

    #[test]
    fn missing_moov_and_truncated_boxes_are_errors() {
        assert!(matches!(
            read_moov(&mut Cursor::new(bx(b"ftyp", b"isom"))),
            Err(Mp4Error::NoMoov)
        ));
        let mut file = bx(b"ftyp", b"isom");
        file.extend_from_slice(&100u32.to_be_bytes());
        file.extend_from_slice(b"moov");
        assert!(matches!(
            read_moov(&mut Cursor::new(file)),
            Err(Mp4Error::Malformed(_))
        ));
    }

    #[test]
    fn sample_tables_expand_with_stco_and_co64_alike() {
        // Chunk 1 holds two samples, chunks 2-3 one each: stsc runs (1, 2) then (2, 1).
        let stsc = [[1, 2, 1], [2, 1, 1]];
        for co64 in [false, true] {
            let offsets: [u64; 3] = if co64 {
                [5_000_000_000, 6_000_000_000, 7_000_000_000]
            } else {
                [100, 500, 900]
            };
            let file = bx(
                b"moov",
                &trak(&spec(&[10, 20, 30, 40], &offsets, &stsc, co64)),
            );
            let samples = read_video_tracks(&mut Cursor::new(file))
                .unwrap()
                .remove(0)
                .samples;
            let got: Vec<(u64, u32, i64, i64, bool)> = samples
                .iter()
                .map(|s| (s.offset, s.size, s.dts, s.pts, s.is_sync))
                .collect();
            assert_eq!(
                got,
                vec![
                    (offsets[0], 10, 0, -1001, true),
                    (offsets[0] + 10, 20, 1001, 0, false),
                    (offsets[1], 30, 2002, 3003, true),
                    (offsets[2], 40, 3003, 4004, false),
                ]
            );
        }
    }

    /// An `stbl` payload: `stsz` (fixed size, count, per-sample sizes), `stts` rows, `stsc` rows and `co64` offsets.
    fn stbl(
        fixed: u32,
        count: u32,
        sizes: &[u32],
        stts: &[[u32; 2]],
        stsc: &[[u32; 3]],
        co64: &[u64],
    ) -> Vec<u8> {
        let mut stsz = fixed.to_be_bytes().to_vec();
        stsz.extend_from_slice(&count.to_be_bytes());
        sizes
            .iter()
            .for_each(|s| stsz.extend_from_slice(&s.to_be_bytes()));
        let rows = |n: usize, cells: Vec<u32>| {
            let mut b = (n as u32).to_be_bytes().to_vec();
            cells
                .iter()
                .for_each(|c| b.extend_from_slice(&c.to_be_bytes()));
            b
        };
        let mut offsets = (co64.len() as u32).to_be_bytes().to_vec();
        co64.iter()
            .for_each(|o| offsets.extend_from_slice(&o.to_be_bytes()));
        let mut out = full(b"stsz", 0, &stsz);
        out.extend(full(b"stts", 0, &rows(stts.len(), stts.concat())));
        out.extend(full(b"stsc", 0, &rows(stsc.len(), stsc.concat())));
        out.extend(full(b"co64", 0, &offsets));
        out
    }

    #[test]
    fn sample_counts_beyond_the_box_or_file_size_are_errors_before_allocation() {
        // A fixed sample size with 2^32 - 1 samples in a 1 MB file: 16 GB of sizes before, an error now.
        let huge = stbl(
            1,
            u32::MAX,
            &[],
            &[[u32::MAX, 1]],
            &[[1, u32::MAX, 1]],
            &[0],
        );
        assert!(matches!(
            sample_table(&huge, 0, 1 << 20),
            Err(Mp4Error::Malformed(_))
        ));
        // Per-sample sizes: 2^32 - 1 entries stated in a box that holds none.
        let short = stbl(0, u32::MAX, &[], &[[u32::MAX, 1]], &[[1, 1, 1]], &[0]);
        assert!(matches!(
            sample_table(&short, 0, 1 << 20),
            Err(Mp4Error::Malformed(_))
        ));
        // One stts entry of 2^32 - 1 samples for a 2-sample table: 32 GB of decode times before, an error now.
        let stts = stbl(0, 2, &[10, 20], &[[u32::MAX, 1]], &[[1, 2, 1]], &[100]);
        assert!(matches!(
            sample_table(&stts, 0, 1 << 20),
            Err(Mp4Error::Malformed(_))
        ));
        // The same table with a matching stts still reads.
        let ok = stbl(0, 2, &[10, 20], &[[2, 1]], &[[1, 2, 1]], &[100]);
        let samples = sample_table(&ok, 0, 1 << 20).unwrap();
        assert_eq!((samples[1].offset, samples[1].dts), (110, 1));
        let fixed = stbl(7, 2, &[], &[[2, 1]], &[[1, 2, 1]], &[100]);
        assert_eq!(sample_table(&fixed, 0, 1 << 20).unwrap()[1].offset, 107);
    }

    #[test]
    fn overflowing_table_values_are_errors_not_panics() {
        // A second stsc run that starts at chunk 0.
        let chunk0 = stbl(
            0,
            2,
            &[10, 20],
            &[[2, 1]],
            &[[1, 1, 1], [0, 1, 1]],
            &[100, 200],
        );
        assert!(matches!(
            sample_table(&chunk0, 0, 1 << 20),
            Err(Mp4Error::Malformed(_))
        ));
        // A chunk offset 5 bytes below 2^64 followed by a 10-byte sample.
        let offset = stbl(0, 2, &[10, 20], &[[2, 1]], &[[1, 2, 1]], &[u64::MAX - 5]);
        assert!(matches!(
            sample_table(&offset, 0, 1 << 20),
            Err(Mp4Error::Malformed(_))
        ));
        // A 64-bit edit-list media time that puts the presentation time past i64.
        let ok = stbl(0, 2, &[10, 20], &[[2, 1]], &[[1, 2, 1]], &[100]);
        assert!(matches!(
            sample_table(&ok, i64::MIN, 1 << 20),
            Err(Mp4Error::Malformed(_))
        ));
    }

    #[test]
    fn a_table_that_disagrees_with_stsz_is_an_error() {
        let file = bx(
            b"moov",
            &trak(&spec(
                &[10, 20, 30, 40, 50],
                &[100, 200],
                &[[1, 2, 1]],
                false,
            )),
        );
        assert!(matches!(
            read_video_tracks(&mut Cursor::new(file)),
            Err(Mp4Error::Malformed(_))
        ));
    }
}
