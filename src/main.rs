// Copyright (C) 2021 Scott Lamb <slamb@slamb.org>
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command-line application to record MP4 video from RTSP cameras.
//!
//! Documentation and repository at
//! [github.com/strawlab/retina-record](https://github.com/strawlab/retina-record).

mod mp4;

use anyhow::Error;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(version, about)]
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
    let opts = mp4::Opts::parse();
    mp4::run(opts).await
}
