// Copyright (C) 2021 Scott Lamb <slamb@slamb.org>
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RTSP client examples.

mod info;
mod mp4;
mod onvif;

use anyhow::Error;
use clap::Parser;

#[derive(Parser)]
struct Source {
    /// `rtsp://` URL to connect to.
    #[clap(long)]
    url: url::Url,

    /// Username to send if the server requires authentication.
    #[clap(long)]
    username: Option<String>,

    /// Password; requires username.
    #[clap(long, requires = "username")]
    password: Option<String>,
}

#[derive(Parser)]
enum Cmd {
    /// Gets info about available streams and exits.
    Info(info::Opts),
    /// Writes available audio and video streams to mp4 file; use Ctrl+C to stop.
    Mp4(mp4::Opts),
    /// Follows ONVIF metadata stream; use Ctrl+C to stop.
    Onvif(onvif::Opts),
}

/// Interpets the `username` and `password` of a [Source].
fn creds(
    username: Option<String>,
    password: Option<String>,
) -> Option<retina::client::Credentials> {
    match (username, password) {
        (Some(username), password) => Some(retina::client::Credentials {
            username,
            password: password.unwrap_or_default(),
        }),
        (None, None) => None,
        _ => unreachable!(), // clap enforces that password requires username.
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt::init();
    let cmd = Cmd::parse();
    match cmd {
        Cmd::Info(opts) => info::run(opts).await,
        Cmd::Mp4(opts) => mp4::run(opts).await,
        Cmd::Onvif(opts) => onvif::run(opts).await,
    }
}
