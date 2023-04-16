use sdl2::{video::Window, Sdl};

use crate::{
    gfx::wgpu::{Wgpu, WgpuError},
    math::{Vector2, Vector3, Vector4},
};

pub mod mesh;
pub mod texture;
mod wgpu;

pub use mesh::*;
pub use texture::*;

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

        let mut wgpu = Wgpu::new(&window, opts.window_size)?;

        let mut mesh = wgpu.pop_mesh();

        mesh.push_vertices(&[
            Vertex {
                position: Vector3::new(0.0, 0.5, 1.0),
                ..Default::default()
            },
            Vertex {
                position: Vector3::new(0.5, -0.5, 1.0),
                ..Default::default()
            },
            Vertex {
                position: Vector3::new(-0.5, -0.5, 1.0),
                ..Default::default()
            },
        ]);

        wgpu.queue_mesh_upload(mesh, 0);

        Ok(Self { wgpu, window })
    }

    #[inline]
    pub fn tick(&mut self) {
        self.wgpu.tick();
        self.wgpu.render_frame();
    }
}
