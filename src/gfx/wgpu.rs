use futures::executor;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, Limits, Queue, RequestAdapterOptions,
    RequestDeviceError,
};

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error(transparent)]
    CannotFindAdapter { source: anyhow::Error },

    #[error(transparent)]
    UnsupportedDevice {
        #[from]
        source: RequestDeviceError,
    },
}

pub struct Wgpu {
    device: Device,
    queue: Queue,
}

impl Wgpu {
    pub fn new<W: HasRawWindowHandle + HasRawDisplayHandle>(window: &W) -> Result<Self, WgpuError> {
        let instance = Instance::new(Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };

        let adapter = executor::block_on(instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..RequestAdapterOptions::default()
        }))
        .ok_or_else(|| WgpuError::CannotFindAdapter {
            source: anyhow::anyhow!("failed to locate a suitable graphics adapter"),
        })?;

        let info = adapter.get_info();
        log::debug!(
            "Using adapter: \"{}\" with {:?} backend",
            info.name,
            info.backend
        );
        log::debug!("Adapter features: {:?}", adapter.features());
        log::debug!("Adapter limits: {:?}", adapter.limits());

        let (device, queue) = executor::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                features: Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | Features::TEXTURE_COMPRESSION_BC,
                limits: Limits::default(),
            },
            None,
        ))?;

        Ok(Self { device, queue })
    }
}
