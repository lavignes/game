use game::gfx::{Gfx, GfxInitOptions};
use log::LevelFilter;

fn main() {
    env_logger::builder()
        .filter_module("game", LevelFilter::Debug)
        .init();

    let sdl = sdl2::init().unwrap();
    let gfx = Gfx::new(&sdl, GfxInitOptions::default());
}
