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
* Stores additional timing data inline in the .mp4 file. First, each frame is
  stored with its "RTP timestamp" and the "receive timestamp". Secondly,
  occasional (every few seconds) sender reports from the camera with additional
  NTP timestamps are stored alongside RTP and receive timestamps. Together, this
  allows good reconstruction of the timing of individual cameras or
  synchronizing videos from multiple cameras.
* Based on the [`retina`](https://crates.io/crates/retina) Rust crate to support
  H264 RTSP cameras. This is the same library underlying [Moonfire
  NVR](https://github.com/scottlamb/moonfire-nvr).
* Written in pure Rust.

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
