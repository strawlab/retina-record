# retina-record

[![Crates.io](https://img.shields.io/crates/v/retina-record)](https://crates.io/crates/retina-record)
[![Crate License](https://img.shields.io/crates/l/retina-record.svg)](https://crates.io/crates/retina-record)
[![Rust](https://github.com/strawlab/retina-record/workflows/Rust/badge.svg)](https://github.com/strawlab/retina-record/actions)

Command-line application to record MP4 video from RTSP cameras.

Documentation and repository at
[github.com/strawlab/retina-record](https://github.com/strawlab/retina-record).

## Features

* Does not transcode video from the camera but streams the already-encoded H264
  video directly to an .mp4 file. Consequently, CPU usage is minimal.
* Stores additional timing data inline in the .mp4 file. See "Timing metadata"
  below.
* Based on the [`retina`](https://crates.io/crates/retina) Rust crate to support
  H264 RTSP cameras. This is the same library underlying [Moonfire
  NVR](https://github.com/scottlamb/moonfire-nvr).
* Written in pure Rust.

## Timing data

`retina-record` saves additional timing data during recording. This allows
best-effort reconstruction of the timing of individual camera frames and
synchronizing videos from multiple cameras. The overall philosophy is to log, in
a lossless manner, the timing information sent from the camera as well as the
received time at which `retina-record` received the data. If the camera has a
good internal clock (e.g. because it is using a well behaved NTP client), the
timestamps from the camera alone should be sufficient to reconstruct quite
precisely (see below for details about "quite precisely") when images were acquired. Alternatively, if
the camera's internal clock is not as ideal, the timestamps saved by
`retina-record` can be used to roughly align the data under the assumptions that
network delays are insignificant and that the clock of the PC on which
`retina-record` runs is reliable.

### Frame timing data

Each frame is stored with its "RTP timestamp" from the camera and the
"receive timestamp" from `retina-record`. The formal description of the data
saved in the H264 stream as Supplemental Enhancement Information (SEI) is at
[strawlab.org/89H](https://strawlab.org/89H/).

### Sender report timing data

Occasional (every few seconds) sender reports from the camera with additional
NTP timestamps are stored alongside RTP and receive timestamps. The formal
description of the data saved in the H264 stream as Supplemental Enhancement
Information (SEI) is at [strawlab.org/o8B](https://strawlab.org/o8B/).

The NTP timestamps are in the 64-bit NTP timestamp format and thus theoretically
have nanosecond-level precision. Practically, this depends on the quality of the
internal clock on the camera.

## Installing

1. [Install rust](https://www.rust-lang.org/tools/install)
2. Run `cargo install retina-record --locked`.

## Running

Here is an example command-line to record from a camera to the file
`output.mp4`. This will run until Ctrl-C is used to stop the program. You will
need to update the URL for your camera and presumably the password and output
filename.

```shell
retina-record mp4 --url rtsp://192.168.1.32/cam1/onvif-h264 --username admin --password secret-password output.mp4
```
