use sdl2::{
    video::{Window, WindowBuildError},
    Sdl,
};

use crate::{
    gfx::wgpu::{Wgpu, WgpuError},
    math::Vector2,
};

mod wgpu;

#[derive(Debug, thiserror::Error)]
pub enum GfxError {
    #[error(transparent)]
    WgpuError {
        #[from]
        source: WgpuError,
    },

    #[error(transparent)]
    SdlError { source: anyhow::Error },
}

#[derive(Debug)]
pub struct GfxInitOptions<'a> {
    window_title: &'a str,
    window_size: Vector2,
}

impl<'a> Default for GfxInitOptions<'a> {
    fn default() -> Self {
        Self {
            window_title: "game",
            window_size: Vector2::new(800.0, 600.0),
        }
    }
}

pub struct Gfx {
    wgpu: Wgpu,
    window: Window,
}

impl Gfx {
    pub fn new(sdl: &Sdl, opts: GfxInitOptions<'_>) -> Result<Self, GfxError> {
        let sdl_video = sdl.video().map_err(|e| GfxError::SdlError {
            source: anyhow::anyhow!(e),
        })?;
        let size: (u32, u32) = opts.window_size.into();
        let window = sdl_video
            .window(opts.window_title, size.0, size.1)
            .position_centered()
            .allow_highdpi()
            .build()
            .map_err(|e| GfxError::SdlError {
                source: anyhow::anyhow!(e),
            })?;

        let wgpu = Wgpu::new(&window)?;

        Ok(Self { wgpu, window })
    }
}
