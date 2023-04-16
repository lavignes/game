use std::time::{Duration, Instant};

use game::gfx::{Gfx, GfxInitOptions};
use log::LevelFilter;
use sdl2::event::Event;

fn main() {
    env_logger::builder()
        .filter_module("game", LevelFilter::Debug)
        .init();

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
            log::debug!("{fps} fps");
            frames = 0;
            t0 = t1;
        }
    }
}
