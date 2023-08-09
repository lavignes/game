use sdl2::{video::Window, Sdl};

use crate::{
    gfx::wgpu::{Wgpu, WgpuError},
    math::{Matrix4, Quaternion, Vector2, Vector3, Vector4},
    util,
};

pub mod mesh;
pub mod texture;
mod wgpu;

pub use mesh::*;
pub use texture::*;

use self::wgpu::WgpuInitOptions;

#[derive(Copy, Clone, Default, Debug)]
pub struct PerspectiveProjection {
    pub fov: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl From<PerspectiveProjection> for Matrix4 {
    #[inline]
    fn from(p: PerspectiveProjection) -> Matrix4 {
        Matrix4::perspective(p.fov, p.aspect_ratio, p.near, p.far)
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Camera {
    position: Vector3,
    euler_angles: Vector2,
}

#[derive(Copy, Clone, Debug)]
pub struct Transform {
    pub position: Vector3,
    pub scale: Vector3,
    pub rotation: Quaternion,
}

impl Transform {
    #[inline]
    pub fn concat(&self, rhs: &Transform) -> Transform {
        Transform {
            position: self.position + rhs.position,
            scale: self.scale * rhs.scale,
            rotation: self.rotation * rhs.rotation,
        }
    }
}

impl Default for Transform {
    #[inline]
    fn default() -> Transform {
        Transform {
            position: Vector3::splat(0.0),
            scale: Vector3::splat(1.0),
            rotation: Quaternion::identity(),
        }
    }
}

impl From<&Transform> for Matrix4 {
    #[inline]
    fn from(t: &Transform) -> Matrix4 {
        &(&Matrix4::scale(t.scale) * &t.rotation.normalized().into())
            * &Matrix4::translate(t.position)
    }
}

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

    projection: PerspectiveProjection,
    camera: Camera,
}

impl<'a> Default for GfxInitOptions<'a> {
    fn default() -> Self {
        Self {
            window_title: "game",
            window_size: Vector2::new(800.0, 600.0),
            projection: PerspectiveProjection {
                fov: 1.0,
                aspect_ratio: 800.0 / 600.0,
                near: 0.001,
                far: 65535.0,
            },
            camera: Camera {
                position: Vector3::new(0.0, 0.0, 16.0),
                euler_angles: Vector2::splat(0.0),
            },
        }
    }
}

pub struct Gfx {
    wgpu: Wgpu,
    window: Window,

    projection: PerspectiveProjection,
    camera: Camera,
}

impl Gfx {
    pub fn new(sdl: &Sdl, opts: GfxInitOptions<'_>) -> Result<Self, GfxError> {
        let GfxInitOptions {
            window_title,
            window_size,
            projection,
            camera,
        } = opts;

        let sdl_video = sdl.video().map_err(|e| GfxError::SdlError {
            source: anyhow::anyhow!(e),
        })?;
        let size: (u32, u32) = window_size.into();
        let window = sdl_video
            .window(window_title, size.0, size.1)
            .position_centered()
            .allow_highdpi()
            .build()
            .map_err(|e| GfxError::SdlError {
                source: anyhow::anyhow!(e),
            })?;

        let mut wgpu = Wgpu::new(
            &window,
            WgpuInitOptions {
                window_size,
                projection,
                camera,
            },
        )?;

        let mut mesh = wgpu.pop_mesh();

        mesh.push_vertices(&[
            Vertex {
                position: Vector3::new(0.0, 0.5, 0.0) * 2.0,
                tex_coord: Vector2::new(0.5, 0.0),
                ..Default::default()
            },
            Vertex {
                position: Vector3::new(0.5, -0.5, 0.0) * 2.0,
                tex_coord: Vector2::new(1.0, 1.0),
                ..Default::default()
            },
            Vertex {
                position: Vector3::new(-0.5, -0.5, 0.0) * 2.0,
                tex_coord: Vector2::new(0.0, 1.0),
                ..Default::default()
            },
        ]);

        wgpu.queue_mesh_upload(mesh, 0);

        let mut texture = wgpu.pop_texture();
        let mut reader = DDSReader::new();

        reader
            .read_into(
                &mut util::buf_open("res/tex/test.dds").unwrap(),
                &mut texture,
            )
            .unwrap();

        wgpu.queue_texture_upload(texture, 0);

        Ok(Self {
            wgpu,
            window,
            projection,
            camera,
        })
    }

    #[inline]
    pub fn tick(&mut self) {
        //self.camera.euler_angles = self.camera.euler_angles + Vector2::new(0.0, 0.2);
        self.wgpu.set_camera(self.camera);
        self.wgpu.set_projection(self.projection);
        self.wgpu.tick();
        self.wgpu.render_frame();
    }
}
