// Copyright (C) 2021 Scott Lamb <slamb@slamb.org>
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `.mp4` writer

use anyhow::{anyhow, bail, Context, Error};
use bytes::{Buf, BufMut, BytesMut};
use clap::Parser;
use futures::{Future, StreamExt};
use retina::{
    client::{SetupOptions, Transport},
    codec::{AudioParameters, CodecItem, ParametersRef, VideoParameters},
};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::{convert::TryFrom, pin::Pin};
use std::{io::SeekFrom, sync::Arc};
use tokio::{
    fs::File,
    io::{AsyncSeek, AsyncSeekExt, AsyncWrite, AsyncWriteExt},
};

#[derive(Parser)]
pub struct Opts {
    #[command(flatten)]
    src: super::Source,

    /// Policy for handling the `rtptime` parameter normally seem in the `RTP-Info` header.
    /// One of `default`, `require`, `ignore`, `permissive`.
    #[arg(default_value_t, long)]
    initial_timestamp: retina::client::InitialTimestampPolicy,

    /// Policy for handling unknown ssrcs in RTCP packets.
    #[arg(default_value_t, long)]
    unknown_rtcp_ssrc: retina::client::UnknownRtcpSsrcPolicy,

    /// Don't attempt to include video streams.
    #[arg(long)]
    no_video: bool,

    /// Don't attempt to include audio streams.
    #[arg(long)]
    no_audio: bool,

    /// Don't include supplemental video timing data.
    #[arg(long)]
    no_timing_info: bool,

    /// Allow lost packets mid-stream without aborting.
    #[arg(long)]
    allow_loss: bool,

    /// When to issue a `TEARDOWN` request: `auto`, `always`, or `never`.
    #[arg(default_value_t, long)]
    teardown: retina::client::TeardownPolicy,

    /// Duration after which to exit automatically, in seconds.
    #[arg(long, name = "secs")]
    duration: Option<u64>,

    /// The transport to use: `tcp` or `udp` (experimental).
    ///
    /// Note: `--allow-loss` is strongly recommended with `udp`.
    #[arg(default_value_t, long)]
    transport: retina::client::Transport,

    /// Path to `.mp4` file to write.
    out: PathBuf,
}

fn chrono_to_ntp<TZ>(
    orig: chrono::DateTime<TZ>,
) -> Result<retina::NtpTimestamp, std::num::TryFromIntError>
where
    TZ: chrono::TimeZone,
{
    let epoch: chrono::DateTime<chrono::Utc> = "1900-01-01 00:00:00Z".parse().unwrap();
    let elapsed: chrono::TimeDelta = orig.to_utc() - epoch;
    let sec_since_epoch: u32 = elapsed.num_seconds().try_into()?;
    let nanos = elapsed.subsec_nanos();
    let frac = nanos as f64 / 1e9;
    let frac_int = (frac * f64::from(u32::MAX)).round() as u32;
    let val = (u64::from(sec_since_epoch) << 32) + u64::from(frac_int);
    Ok(retina::NtpTimestamp(val))
}

#[test]
fn test_ntp_roundtrip() {
    let orig_str = "2024-02-17T21:14:34.013+01:00";
    let orig: chrono::DateTime<chrono::Utc> = orig_str.parse().unwrap();
    let ntp_timestamp = chrono_to_ntp(orig).unwrap();
    let display = format!("{ntp_timestamp}");
    let parsed: chrono::DateTime<chrono::Utc> = display.parse().unwrap();
    assert_eq!(orig, parsed);
}

/// Writes a box length for everything appended in the supplied scope.
macro_rules! write_box {
    ($buf:expr, $fourcc:expr, $b:block) => {{
        let _: &mut BytesMut = $buf; // type-check.
        let pos_start = ($buf as &BytesMut).len();
        let fourcc: &[u8; 4] = $fourcc;
        $buf.extend_from_slice(&[0, 0, 0, 0, fourcc[0], fourcc[1], fourcc[2], fourcc[3]]);
        let r = {
            $b;
        };
        let pos_end = ($buf as &BytesMut).len();
        let len = pos_end.checked_sub(pos_start).unwrap();
        $buf[pos_start..pos_start + 4].copy_from_slice(&u32::try_from(len)?.to_be_bytes()[..]);
        r
    }};
}

/// Writes `.mp4` data to a sink.
/// See module-level documentation for details.
pub struct Mp4Writer<W: AsyncWrite + AsyncSeek + Send + Unpin> {
    mdat_start: u64,
    mdat_pos: u64,
    video_params: Vec<VideoParameters>,

    /// The most recently used 1-based index within `video_params`.
    cur_video_params_sample_description_index: Option<u32>,
    audio_params: Option<Box<AudioParameters>>,
    allow_loss: bool,

    /// The (1-indexed) video sample (frame) number of each sync sample (random access point).
    video_sync_sample_nums: Vec<u32>,

    save_timing_info: bool,
    sender_report: Option<SenderReportInfo>,

    video_trak: TrakTracker,
    audio_trak: TrakTracker,
    inner: W,
}

/// Timing information received occasionally via RTCP sender reports
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SenderReportInfo {
    /// Receive timestamp as NTP (Network Time Protocol) timestamp
    recv: u64,
    /// NTP (Network Time Protocol) timestamp as reported by the sender
    ntp: u64,
    /// RTP (Real Time Protocol) timestamp as reported by the sender
    rtp: u32,
}

impl SenderReportInfo {
    const fn uuid() -> &'static [u8; 16] {
        b"strawlab.org/o8B"
    }
    fn from_sender_report(orig: retina::rtcp::SenderReportRef<'_>) -> Self {
        let recv = chrono_to_ntp(chrono::Utc::now()).unwrap().0;
        let ntp = orig.ntp_timestamp().0;
        let rtp = orig.rtp_timestamp();
        Self { recv, ntp, rtp }
    }
}

/// Timing information associated with each video frame
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FrameInfo {
    /// Receive timestamp as NTP (Network Time Protocol) timestamp
    recv: u64,
    /// RTP (Real Time Protocol) timestamp as reported by the sender
    rtp: u32,
}

impl FrameInfo {
    const fn uuid() -> &'static [u8; 16] {
        b"strawlab.org/89H"
    }
    fn new(timestamp: retina::Timestamp) -> Self {
        let recv = chrono_to_ntp(chrono::Utc::now()).unwrap().0;
        let rtp = timestamp.timestamp() as u32;
        Self { recv, rtp }
    }
}

fn udu_to_rbsp(uuid: &[u8; 16], buf: &[u8]) -> Vec<u8> {
    let size = uuid.len() + buf.len();
    let n255s = size / 256;
    let rem: u8 = (size % 256) as u8;

    let size_buf = {
        let mut size_buf = vec![0xff; n255s + 1];
        size_buf[n255s] = rem;
        size_buf
    };

    // uuid_iso_iec_11578

    let final_size = 3 + size_buf.len() + size;
    let mut final_buf = vec![0; final_size];

    final_buf[0] = 0x06; // code 6 - SEI
    final_buf[1] = 0x05; // header type: UserDataUnregistered
    final_buf[2..2 + size_buf.len()].copy_from_slice(&size_buf);
    final_buf[2 + size_buf.len()..2 + size_buf.len() + uuid.len()].copy_from_slice(uuid);
    final_buf[2 + size_buf.len() + uuid.len()..final_size - 1].copy_from_slice(buf);
    final_buf[final_size - 1] = 0x80;
    final_buf
}

/// A chunk: a group of samples that have consecutive byte positions and same sample description.
struct Chunk {
    first_sample_number: u32, // 1-based index
    byte_pos: u64,            // starting byte of first sample
    sample_description_index: u32,
}

/// Tracks the parts of a `trak` atom which are common between video and audio samples.
#[derive(Default)]
struct TrakTracker {
    samples: u32,
    next_pos: Option<u64>,
    chunks: Vec<Chunk>,
    sizes: Vec<u32>,

    /// The durations of samples in a run-length encoding form: (number of samples, duration).
    /// This lags one sample behind calls to `add_sample` because each sample's duration
    /// is calculated using the PTS of the following sample.
    durations: Vec<(u32, u32)>,
    last_pts: Option<i64>,
    tot_duration: u64,
}

impl TrakTracker {
    fn add_sample(
        &mut self,
        sample_description_index: u32,
        byte_pos: u64,
        size: u32,
        timestamp: retina::Timestamp,
        loss: u16,
        allow_loss: bool,
    ) -> Result<(), Error> {
        if self.samples > 0 && loss > 0 && !allow_loss {
            bail!("Lost {} RTP packets mid-stream", loss);
        }
        if self.samples > 0 && loss > 0 {
            warn!("Lost {} RTP packets mid-stream", loss);
        }
        self.samples += 1;
        if self.next_pos != Some(byte_pos)
            || self.chunks.last().map(|c| c.sample_description_index)
                != Some(sample_description_index)
        {
            self.chunks.push(Chunk {
                first_sample_number: self.samples,
                byte_pos,
                sample_description_index,
            });
        }
        self.sizes.push(size);
        self.next_pos = Some(byte_pos + u64::from(size));
        if let Some(last_pts) = self.last_pts.replace(timestamp.timestamp()) {
            let duration = timestamp.timestamp().checked_sub(last_pts).unwrap();
            self.tot_duration += u64::try_from(duration).unwrap();
            let duration = u32::try_from(duration)?;
            match self.durations.last_mut() {
                Some((s, d)) if *d == duration => *s += 1,
                _ => self.durations.push((1, duration)),
            }
        }
        Ok(())
    }

    fn finish(&mut self) {
        if self.last_pts.is_some() {
            self.durations.push((1, 0));
        }
    }

    /// Estimates the sum of the variable-sized portions of the data.
    fn size_estimate(&self) -> usize {
        (self.durations.len() * 8) + // stts
        (self.chunks.len() * 12) +   // stsc
        (self.sizes.len() * 4) +     // stsz
        (self.chunks.len() * 4) // stco
    }

    fn write_common_stbl_parts(&self, buf: &mut BytesMut) -> Result<(), Error> {
        // TODO: add an edit list so the video and audio tracks are in sync.
        write_box!(buf, b"stts", {
            buf.put_u32(0);
            buf.put_u32(u32::try_from(self.durations.len())?);
            for (samples, duration) in &self.durations {
                buf.put_u32(*samples);
                buf.put_u32(*duration);
            }
        });
        write_box!(buf, b"stsc", {
            buf.put_u32(0); // version
            buf.put_u32(u32::try_from(self.chunks.len())?);
            let mut prev_sample_number = 1;
            let mut chunk_number = 1;
            if !self.chunks.is_empty() {
                for c in &self.chunks[1..] {
                    buf.put_u32(chunk_number);
                    buf.put_u32(c.first_sample_number - prev_sample_number);
                    buf.put_u32(c.sample_description_index);
                    prev_sample_number = c.first_sample_number;
                    chunk_number += 1;
                }
                buf.put_u32(chunk_number);
                buf.put_u32(self.samples + 1 - prev_sample_number);
                buf.put_u32(1); // sample_description_index
            }
        });
        write_box!(buf, b"stsz", {
            buf.put_u32(0); // version
            buf.put_u32(0); // sample_size
            buf.put_u32(u32::try_from(self.sizes.len())?);
            for s in &self.sizes {
                buf.put_u32(*s);
            }
        });
        write_box!(buf, b"co64", {
            buf.put_u32(0); // version
            buf.put_u32(u32::try_from(self.chunks.len())?); // entry_count
            for c in &self.chunks {
                buf.put_u64(c.byte_pos);
            }
        });
        Ok(())
    }
}

impl<W: AsyncWrite + AsyncSeek + Send + Unpin> Mp4Writer<W> {
    pub async fn new(
        audio_params: Option<Box<AudioParameters>>,
        allow_loss: bool,
        save_timing_info: bool,
        mut inner: W,
    ) -> Result<Self, Error> {
        let mut buf = BytesMut::new();
        write_box!(&mut buf, b"ftyp", {
            buf.extend_from_slice(&[
                b'i', b's', b'o', b'm', // major_brand
                0, 0, 0, 0, // minor_version
                b'i', b's', b'o', b'm', // compatible_brands[0]
            ]);
        });

        let mut mdat_large_header = [0u8; 16];
        mdat_large_header[0..4].copy_from_slice( &1u32.to_be_bytes()[..]);
        mdat_large_header[4..8].copy_from_slice(b"mdat");
        buf.extend_from_slice(&mdat_large_header[..]);
        let mdat_start = u64::try_from(buf.len())?;
        inner.write_all(&buf).await?;
        Ok(Mp4Writer {
            inner,
            video_params: Vec::new(),
            cur_video_params_sample_description_index: None,
            audio_params,
            allow_loss,
            video_trak: TrakTracker::default(),
            audio_trak: TrakTracker::default(),
            video_sync_sample_nums: Vec::new(),
            save_timing_info,
            sender_report: Default::default(),
            mdat_start,
            mdat_pos: mdat_start,
        })
    }

    pub async fn finish(mut self) -> Result<(), Error> {
        self.video_trak.finish();
        self.audio_trak.finish();
        let mut buf = BytesMut::with_capacity(
            1024 + self.video_trak.size_estimate()
                + self.audio_trak.size_estimate()
                + 4 * self.video_sync_sample_nums.len(),
        );
        write_box!(&mut buf, b"moov", {
            write_box!(&mut buf, b"mvhd", {
                buf.put_u32(1 << 24); // version
                buf.put_u64(0); // creation_time
                buf.put_u64(0); // modification_time
                buf.put_u32(90000); // timescale
                buf.put_u64(self.video_trak.tot_duration);
                buf.put_u32(0x00010000); // rate
                buf.put_u16(0x0100); // volume
                buf.put_u16(0); // reserved
                buf.put_u64(0); // reserved
                for v in &[0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000] {
                    buf.put_u32(*v); // matrix
                }
                for _ in 0..6 {
                    buf.put_u32(0); // pre_defined
                }
                buf.put_u32(2); // next_track_id
            });
            if self.video_trak.samples > 0 {
                self.write_video_trak(&mut buf)?;
            }
            if self.audio_trak.samples > 0 {
                self.write_audio_trak(&mut buf, self.audio_params.as_ref().unwrap())?;
            }
        });
        self.inner.write_all(&buf).await?;
        self.inner
            .seek(SeekFrom::Start(self.mdat_start - 8))
            .await?;
        self.inner
            .write_all(&(self.mdat_pos + 16 - self.mdat_start).to_be_bytes()[..])
            .await?;
        Ok(())
    }

    fn write_video_trak(&self, buf: &mut BytesMut) -> Result<(), Error> {
        write_box!(buf, b"trak", {
            write_box!(buf, b"tkhd", {
                buf.put_u32((1 << 24) | 7); // version, flags
                buf.put_u64(0); // creation_time
                buf.put_u64(0); // modification_time
                buf.put_u32(1); // track_id
                buf.put_u32(0); // reserved
                buf.put_u64(self.video_trak.tot_duration);
                buf.put_u64(0); // reserved
                buf.put_u16(0); // layer
                buf.put_u16(0); // alternate_group
                buf.put_u16(0); // volume
                buf.put_u16(0); // reserved
                for v in &[0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000] {
                    buf.put_u32(*v); // matrix
                }
                let dims = self.video_params.iter().fold((0, 0), |prev_dims, p| {
                    let dims = p.pixel_dimensions();
                    (
                        std::cmp::max(prev_dims.0, dims.0),
                        std::cmp::max(prev_dims.1, dims.1),
                    )
                });
                let width = u32::from(u16::try_from(dims.0)?) << 16;
                let height = u32::from(u16::try_from(dims.1)?) << 16;
                buf.put_u32(width);
                buf.put_u32(height);
            });
            write_box!(buf, b"mdia", {
                write_box!(buf, b"mdhd", {
                    buf.put_u32(1 << 24); // version
                    buf.put_u64(0); // creation_time
                    buf.put_u64(0); // modification_time
                    buf.put_u32(90000); // timebase
                    buf.put_u64(self.video_trak.tot_duration);
                    buf.put_u32(0x55c40000); // language=und + pre-defined
                });
                write_box!(buf, b"hdlr", {
                    buf.extend_from_slice(&[
                        0x00, 0x00, 0x00, 0x00, // version + flags
                        0x00, 0x00, 0x00, 0x00, // pre_defined
                        b'v', b'i', b'd', b'e', // handler = vide
                        0x00, 0x00, 0x00, 0x00, // reserved[0]
                        0x00, 0x00, 0x00, 0x00, // reserved[1]
                        0x00, 0x00, 0x00, 0x00, // reserved[2]
                        0x00, // name, zero-terminated (empty)
                    ]);
                });
                write_box!(buf, b"minf", {
                    write_box!(buf, b"vmhd", {
                        buf.put_u32(1);
                        buf.put_u64(0);
                    });
                    write_box!(buf, b"dinf", {
                        write_box!(buf, b"dref", {
                            buf.put_u32(0);
                            buf.put_u32(1); // entry_count
                            write_box!(buf, b"url ", {
                                buf.put_u32(1); // version, flags=self-contained
                            });
                        });
                    });
                    write_box!(buf, b"stbl", {
                        write_box!(buf, b"stsd", {
                            buf.put_u32(0); // version
                            buf.put_u32(u32::try_from(self.video_params.len())?); // entry_count
                            for p in &self.video_params {
                                self.write_video_sample_entry(buf, p)?;
                            }
                        });
                        self.video_trak.write_common_stbl_parts(buf)?;
                        write_box!(buf, b"stss", {
                            buf.put_u32(0); // version
                            buf.put_u32(u32::try_from(self.video_sync_sample_nums.len())?);
                            for n in &self.video_sync_sample_nums {
                                buf.put_u32(*n);
                            }
                        });
                    });
                });
            });
        });
        Ok(())
    }

    fn write_audio_trak(
        &self,
        buf: &mut BytesMut,
        parameters: &AudioParameters,
    ) -> Result<(), Error> {
        write_box!(buf, b"trak", {
            write_box!(buf, b"tkhd", {
                buf.put_u32((1 << 24) | 7); // version, flags
                buf.put_u64(0); // creation_time
                buf.put_u64(0); // modification_time
                buf.put_u32(2); // track_id
                buf.put_u32(0); // reserved
                buf.put_u64(self.audio_trak.tot_duration);
                buf.put_u64(0); // reserved
                buf.put_u16(0); // layer
                buf.put_u16(0); // alternate_group
                buf.put_u16(0); // volume
                buf.put_u16(0); // reserved
                for v in &[0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000] {
                    buf.put_u32(*v); // matrix
                }
                buf.put_u32(0); // width
                buf.put_u32(0); // height
            });
            write_box!(buf, b"mdia", {
                write_box!(buf, b"mdhd", {
                    buf.put_u32(1 << 24); // version
                    buf.put_u64(0); // creation_time
                    buf.put_u64(0); // modification_time
                    buf.put_u32(parameters.clock_rate());
                    buf.put_u64(self.audio_trak.tot_duration);
                    buf.put_u32(0x55c40000); // language=und + pre-defined
                });
                write_box!(buf, b"hdlr", {
                    buf.extend_from_slice(&[
                        0x00, 0x00, 0x00, 0x00, // version + flags
                        0x00, 0x00, 0x00, 0x00, // pre_defined
                        b's', b'o', b'u', b'n', // handler = soun
                        0x00, 0x00, 0x00, 0x00, // reserved[0]
                        0x00, 0x00, 0x00, 0x00, // reserved[1]
                        0x00, 0x00, 0x00, 0x00, // reserved[2]
                        0x00, // name, zero-terminated (empty)
                    ]);
                });
                write_box!(buf, b"minf", {
                    write_box!(buf, b"smhd", {
                        buf.extend_from_slice(&[
                            0x00, 0x00, 0x00, 0x00, // version + flags
                            0x00, 0x00, // balance
                            0x00, 0x00, // reserved
                        ]);
                    });
                    write_box!(buf, b"dinf", {
                        write_box!(buf, b"dref", {
                            buf.put_u32(0);
                            buf.put_u32(1); // entry_count
                            write_box!(buf, b"url ", {
                                buf.put_u32(1); // version, flags=self-contained
                            });
                        });
                    });
                    write_box!(buf, b"stbl", {
                        write_box!(buf, b"stsd", {
                            buf.put_u32(0); // version
                            buf.put_u32(1); // entry_count
                            buf.extend_from_slice(
                                parameters
                                    .sample_entry()
                                    .expect("all added streams have sample entries"),
                            );
                        });
                        self.audio_trak.write_common_stbl_parts(buf)?;

                        // AAC requires two samples (really, each is a set of 960 or 1024 samples)
                        // to decode accurately. See
                        // https://developer.apple.com/library/archive/documentation/QuickTime/QTFF/QTFFAppenG/QTFFAppenG.html .
                        write_box!(buf, b"sgpd", {
                            // BMFF section 8.9.3: SampleGroupDescriptionBox
                            buf.put_u32(0); // version
                            buf.extend_from_slice(b"roll"); // grouping type
                            buf.put_u32(1); // entry_count
                                            // BMFF section 10.1: AudioRollRecoveryEntry
                            buf.put_i16(-1); // roll_distance
                        });
                        write_box!(buf, b"sbgp", {
                            // BMFF section 8.9.2: SampleToGroupBox
                            buf.put_u32(0); // version
                            buf.extend_from_slice(b"roll"); // grouping type
                            buf.put_u32(1); // entry_count
                            buf.put_u32(self.audio_trak.samples);
                            buf.put_u32(1); // group_description_index
                        });
                    });
                });
            });
        });
        Ok(())
    }

    fn write_video_sample_entry(
        &self,
        buf: &mut BytesMut,
        parameters: &VideoParameters,
    ) -> Result<(), Error> {
        // TODO: this should move to client::VideoParameters::sample_entry() or some such.
        write_box!(buf, b"avc1", {
            buf.put_u32(0);
            buf.put_u32(1); // data_reference_index = 1
            buf.extend_from_slice(&[0; 16]);
            buf.put_u16(u16::try_from(parameters.pixel_dimensions().0)?);
            buf.put_u16(u16::try_from(parameters.pixel_dimensions().1)?);
            buf.extend_from_slice(&[
                0x00, 0x48, 0x00, 0x00, // horizresolution
                0x00, 0x48, 0x00, 0x00, // vertresolution
                0x00, 0x00, 0x00, 0x00, // reserved
                0x00, 0x01, // frame count
                0x00, 0x00, 0x00, 0x00, // compressorname
                0x00, 0x00, 0x00, 0x00, //
                0x00, 0x00, 0x00, 0x00, //
                0x00, 0x00, 0x00, 0x00, //
                0x00, 0x00, 0x00, 0x00, //
                0x00, 0x00, 0x00, 0x00, //
                0x00, 0x00, 0x00, 0x00, //
                0x00, 0x00, 0x00, 0x00, //
                0x00, 0x18, 0xff, 0xff, // depth + pre_defined
            ]);
            write_box!(buf, b"avcC", {
                buf.extend_from_slice(parameters.extra_data());
            });
        });
        Ok(())
    }

    fn enqueue_sender_report(
        &mut self,
        sr: retina::rtcp::SenderReportRef<'_>,
    ) -> Result<(), Error> {
        let sri = SenderReportInfo::from_sender_report(sr);

        if self.sender_report.replace(sri).is_some() {
            todo!("service report already present. return non-fatal error.")
        }
        Ok(())
    }

    async fn video(
        &mut self,
        stream: &retina::client::Stream,
        frame: retina::codec::VideoFrame,
    ) -> Result<(), Error> {
        let timing_info_buf = if self.save_timing_info {
            // For every frame, we write additional timing information.
            let mut buf = {
                let json_buf = serde_json::to_vec(&FrameInfo::new(frame.timestamp())).unwrap();
                let rbsp_buf = udu_to_rbsp(FrameInfo::uuid(), &json_buf);
                // Because RBSP is JSON and header, emulation protection is not
                // needed because there are no consequtive null bytes. Thus, the
                // RBSP is the EBSP for this case.
                let ebsp_buf = rbsp_buf;
                buf_to_avcc(&ebsp_buf)
            };

            // When we have recieved a service report, write that information, too.
            if let Some(sr) = self.sender_report.take() {
                let json_buf = serde_json::to_vec(&sr).unwrap();
                let rbsp_buf = udu_to_rbsp(SenderReportInfo::uuid(), &json_buf);
                // Because RBSP is JSON and header, emulation protection is not
                // needed because there are no consequtive null bytes. Thus, the
                // RBSP is the EBSP for this case.
                let ebsp_buf = rbsp_buf;
                let head = buf_to_avcc(&ebsp_buf);

                // Make the service report buffer come first as we received it
                // first. Then extend the buffer with the frame info.
                let tail = std::mem::replace(&mut buf, head);
                buf.extend(tail);
            };
            buf
        } else {
            vec![]
        };

        tracing::trace!(
            "{}: {}-byte video frame",
            &frame.timestamp(),
            frame.data().remaining(),
        );
        let sample_description_index = if let (Some(i), false) = (
            self.cur_video_params_sample_description_index,
            frame.has_new_parameters(),
        ) {
            // Use the most recent sample description index for most frames, without having to
            // scan through self.video_sample_index.
            i
        } else {
            match stream.parameters() {
                Some(ParametersRef::Video(params)) => {
                    tracing::info!("new video params: {:?}", params);
                    let pos = self.video_params.iter().position(|p| p == params);
                    if let Some(pos) = pos {
                        u32::try_from(pos + 1)?
                    } else {
                        self.video_params.push(params.clone());
                        u32::try_from(self.video_params.len())?
                    }
                }
                None => {
                    debug!("Discarding video frame received before parameters");
                    return Ok(());
                }
                _ => unreachable!(),
            }
        };
        self.cur_video_params_sample_description_index = Some(sample_description_index);
        let size = u32::try_from(timing_info_buf.len() + frame.data().remaining())?;
        self.video_trak.add_sample(
            sample_description_index,
            self.mdat_pos,
            size,
            frame.timestamp(),
            frame.loss(),
            self.allow_loss,
        )?;
        self.mdat_pos = self
            .mdat_pos
            .checked_add(u64::from(size))
            .ok_or_else(|| anyhow!("mdat_pos overflow"))?;
        if frame.is_random_access_point() {
            self.video_sync_sample_nums.push(self.video_trak.samples);
        }
        if !timing_info_buf.is_empty() {
            self.inner.write_all(&timing_info_buf).await?;
        }
        self.inner.write_all(frame.data()).await?;
        Ok(())
    }

    async fn audio(&mut self, frame: retina::codec::AudioFrame) -> Result<(), Error> {
        tracing::trace!(
            "{}: {}-byte audio frame",
            frame.timestamp(),
            frame.data().remaining()
        );
        let size = u32::try_from(frame.data().remaining())?;
        self.audio_trak.add_sample(
            /* sample_description_index */ 1,
            self.mdat_pos,
            size,
            frame.timestamp(),
            frame.loss(),
            self.allow_loss,
        )?;
        self.mdat_pos = self
            .mdat_pos
            .checked_add(u64::from(size))
            .ok_or_else(|| anyhow!("mdat_pos overflow"))?;
        self.inner.write_all(frame.data()).await?;
        Ok(())
    }
}

/// Copies packets from `session` to `mp4` without handling any cleanup on error.
async fn copy<'a>(
    opts: &'a Opts,
    session: &'a mut retina::client::Demuxed,
    stop_signal: Pin<Box<dyn Future<Output = Result<(), std::io::Error>>>>,
    mp4: &'a mut Mp4Writer<File>,
) -> Result<(), Error> {
    let sleep = match opts.duration {
        Some(secs) => {
            futures::future::Either::Left(tokio::time::sleep(std::time::Duration::from_secs(secs)))
        }
        None => futures::future::Either::Right(futures::future::pending()),
    };
    tokio::pin!(stop_signal);
    tokio::pin!(sleep);
    loop {
        tokio::select! {
            pkt = session.next() => {
                match pkt.ok_or_else(|| anyhow!("EOF"))?? {
                    CodecItem::VideoFrame(f) => {
                        let stream = &session.streams()[f.stream_id()];
                        let start_ctx = *f.start_ctx();
                        mp4.video(stream, f).await.with_context(
                            || format!("Error processing video frame starting with {start_ctx}"))?;
                    },
                    CodecItem::AudioFrame(f) => {
                        let ctx = *f.ctx();
                        mp4.audio(f).await.with_context(
                            || format!("Error processing audio frame, {ctx}"))?;
                    },
                    CodecItem::Rtcp(rtcp) => {
                        if let (Some(t), Some(Ok(Some(sr)))) = (rtcp.rtp_timestamp(), rtcp.pkts().next().map(retina::rtcp::PacketRef::as_sender_report)) {
                            tracing::debug!("{}: SR ts={}", t, sr.ntp_timestamp());
                            mp4.enqueue_sender_report(sr)?;
                        }
                    },
                    _ => continue,
                };
            },
            _ = &mut stop_signal => {
                info!("Stopping due to signal");
                break;
            },
            _ = &mut sleep => {
                info!("Stopping after {} seconds", opts.duration.unwrap());
                break;
            },
        }
    }
    Ok(())
}

/// Writes the `.mp4`, including trying to finish or clean up the file.
async fn write_mp4(
    opts: &Opts,
    session: retina::client::Session<retina::client::Described>,
    audio_params: Option<Box<AudioParameters>>,
    stop_signal: Pin<Box<dyn Future<Output = Result<(), std::io::Error>>>>,
) -> Result<(), Error> {
    let mut session = session
        .play(
            retina::client::PlayOptions::default()
                .initial_timestamp(opts.initial_timestamp)
                .enforce_timestamps_with_max_jump_secs(NonZeroU32::new(10).unwrap())
                .unknown_rtcp_ssrc(opts.unknown_rtcp_ssrc),
        )
        .await?
        .demuxed()?;

    // Append into a filename suffixed with ".partial", then try to either rename it into
    // place if it's complete or delete it otherwise.
    const PARTIAL_SUFFIX: &str = ".partial";
    let mut tmp_filename = opts.out.as_os_str().to_owned();
    tmp_filename.push(PARTIAL_SUFFIX); // OsString::push doesn't put in a '/', unlike PathBuf::.
    let tmp_filename: PathBuf = tmp_filename.into();
    let out = tokio::fs::File::create(&tmp_filename).await?;
    let save_timing_info = !opts.no_timing_info;
    let mut mp4 = Mp4Writer::new(audio_params, opts.allow_loss, save_timing_info, out).await?;
    let result = copy(opts, &mut session, stop_signal, &mut mp4).await;
    if let Err(e) = result {
        // Log errors about finishing, returning the original error.
        if let Err(e) = mp4.finish().await {
            tracing::error!(".mp4 finish failed: {}", e);
            if let Err(e) = tokio::fs::remove_file(&tmp_filename).await {
                tracing::error!("and removing .mp4 failed too: {}", e);
            }
        } else if let Err(e) = tokio::fs::rename(&tmp_filename, &opts.out).await {
            tracing::error!("unable to move completed .mp4 into place: {}", e);
        }
        Err(e)
    } else {
        // Directly return errors about finishing.
        if let Err(e) = mp4.finish().await {
            tracing::error!(".mp4 finish failed: {}", e);
            if let Err(e) = tokio::fs::remove_file(&tmp_filename).await {
                tracing::error!("and removing .mp4 failed too: {}", e);
            }
            Err(e)
        } else {
            tokio::fs::rename(&tmp_filename, &opts.out).await?;
            Ok(())
        }
    }
}

fn buf_to_avcc(nal: &[u8]) -> Vec<u8> {
    let sz: u32 = nal.len().try_into().unwrap();
    let mut result = vec![0u8; nal.len() + 4];
    result[0..4].copy_from_slice(&sz.to_be_bytes());
    result[4..].copy_from_slice(nal);
    result
}

pub async fn run(opts: Opts) -> Result<(), Error> {
    if matches!(opts.transport, Transport::Udp(_)) && !opts.allow_loss {
        warn!("Using --transport=udp without strongly recommended --allow-loss!");
    }

    let creds = super::creds(opts.src.username.clone(), opts.src.password.clone());
    let stop_signal = Box::pin(tokio::signal::ctrl_c());
    let session_group = Arc::new(retina::client::SessionGroup::default());
    let mut session = retina::client::Session::describe(
        opts.src.url.clone(),
        retina::client::SessionOptions::default()
            .creds(creds)
            .session_group(session_group.clone())
            .user_agent("Retina mp4 example".to_owned())
            .teardown(opts.teardown),
    )
    .await?;
    let video_stream_i = if !opts.no_video {
        let s = session.streams().iter().position(|s| {
            if s.media() == "video" {
                if s.encoding_name() == "h264" {
                    tracing::info!("Using h264 video stream");
                    return true;
                }
                tracing::info!(
                    "Ignoring {} video stream because it's unsupported",
                    s.encoding_name(),
                );
            }
            false
        });
        if s.is_none() {
            tracing::info!("No suitable video stream found");
        }
        s
    } else {
        tracing::info!("Ignoring video streams (if any) because of --no-video");
        None
    };
    if let Some(i) = video_stream_i {
        session
            .setup(i, SetupOptions::default().transport(opts.transport.clone()))
            .await?;
    }
    let audio_stream = if !opts.no_audio {
        let s = session
            .streams()
            .iter()
            .enumerate()
            .find_map(|(i, s)| match s.parameters() {
                // Only consider audio streams that can produce a .mp4 sample
                // entry.
                Some(retina::codec::ParametersRef::Audio(a)) if a.sample_entry().is_some() => {
                    tracing::info!("Using {} audio stream (rfc 6381 codec {})", s.encoding_name(), a.rfc6381_codec().unwrap());
                    Some((i, Box::new(a.clone())))
                }
                _ if s.media() == "audio" => {
                    tracing::info!("Ignoring {} audio stream because it can't be placed into a .mp4 file without transcoding", s.encoding_name());
                    None
                }
                _ => None,
            });
        if s.is_none() {
            tracing::info!("No suitable audio stream found");
        }
        s
    } else {
        tracing::info!("Ignoring audio streams (if any) because of --no-audio");
        None
    };
    if let Some((i, _)) = audio_stream {
        session
            .setup(i, SetupOptions::default().transport(opts.transport.clone()))
            .await?;
    }
    if video_stream_i.is_none() && audio_stream.is_none() {
        bail!("Exiting because no video or audio stream was selected; see info log messages above");
    }
    let result = write_mp4(&opts, session, audio_stream.map(|(_i, p)| p), stop_signal).await;
    if result.is_err() {
        tracing::error!("Writing MP4 failed. Taking down RTSP session.");
    }

    // Session has now been dropped, on success or failure. A TEARDOWN should
    // be pending if necessary. session_group.await_teardown() will wait for it.
    if let Err(e) = session_group.await_teardown().await {
        tracing::error!("TEARDOWN failed: {}", e);
    }
    result
}
