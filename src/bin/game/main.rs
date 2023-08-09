use std::{
    io,
    time::{Duration, Instant},
};

use clap::Parser;
use game::gfx::{Gfx, GfxInitOptions};
use sdl2::event::Event;
use tracing::metadata::LevelFilter;
use tracing_subscriber::{
    filter::{filter_fn, FilterExt},
    prelude::*,
    util::SubscriberInitExt,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// One of `TRACE`, `DEBUG`, `INFO`, `WARN`, or `ERROR`
    #[arg(short, long, default_value_t = LevelFilter::DEBUG)]
    log_level: LevelFilter,
}

fn main() {
    let args = Args::parse();
    let subscriber = tracing_subscriber::fmt::layer()
        .with_thread_names(true)
        .with_writer(io::stderr)
        .with_filter(
            args.log_level
                .and(filter_fn(|meta| meta.target().starts_with("game"))),
        );
    tracing_subscriber::registry().with(subscriber).init();

    let sdl = sdl2::init().unwrap();
    let mut gfx = Gfx::new(&sdl, GfxInitOptions::default()).unwrap();

    let mut event_pump = sdl.event_pump().unwrap();

    let mut t0 = Instant::now();
    let mut frames = 0;
    'running: loop {
        // TODO: Input subsystem
        while let Some(event) = event_pump.poll_event() {
            match event {
                Event::Quit { .. } => break 'running,
                _ => {}
            }
        }

        gfx.tick();

        frames += 1;
        let t1 = Instant::now();
        let dt = t1.duration_since(t0);
        if dt > Duration::from_secs(5) {
            let fps = frames / dt.as_secs();
            tracing::debug!("{fps} fps");
            frames = 0;
            t0 = t1;
        }
    }
}
