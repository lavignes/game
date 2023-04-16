use std::{cmp::Ordering, collections::BinaryHeap, mem, ops::Range};

use fnv::{FnvHashMap, FnvHashSet};
use futures::executor;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{
    AdapterInfo, Backends, BlendState, Buffer, BufferAddress, BufferDescriptor, BufferUsages,
    Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, Device, DeviceDescriptor,
    Extent3d, Features, FragmentState, FrontFace, IndexFormat, Instance, InstanceDescriptor,
    Limits, LoadOp, MultisampleState, Operations, PipelineLayoutDescriptor, PolygonMode,
    PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, Surface, SurfaceConfiguration,
    Texture as WgpuTexture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor, VertexBufferLayout, VertexState, VertexStepMode,
};

use crate::{
    gfx::{Mesh, Texture, Vertex},
    math::Vector2,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MeshId(usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct InstanceId(usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TextureId(usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum ShaderId {
    Mesh,
}

enum UploadJob {
    Mesh {
        priority: usize,
        id: MeshId,
        src_vertex_offset: usize,
        src_index_offset: usize,
        dst_vertex_offset: usize,
        dst_index_offset: usize,
    },
    Texture {
        priority: usize,
        id: TextureId,
        src_offset: usize,
        dst_layer: usize,
    },
}

impl UploadJob {
    #[inline]
    fn priority(&self) -> usize {
        match self {
            Self::Mesh { priority, .. } => *priority,
            Self::Texture { priority, .. } => *priority,
        }
    }
}

impl Ord for UploadJob {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority().cmp(&other.priority())
    }
}

impl PartialOrd for UploadJob {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for UploadJob {}

impl PartialEq for UploadJob {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.priority() == other.priority()
    }
}

struct VertexBuffer {
    buffer: Buffer,
    staging_buffer: Buffer,
    freelist: Vec<Range<usize>>,
    offsets: FnvHashMap<MeshId, Range<usize>>,
}

impl VertexBuffer {
    #[inline]
    fn new(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("vertex uberbuffer"),
            size: 1024 * 1024,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("vertex staging buffer"),
            size: 4 * 1024,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });
        Self {
            staging_buffer,
            freelist: vec![0..buffer.size() as usize],
            offsets: FnvHashMap::default(),
            buffer,
        }
    }
}

struct IndexBuffer {
    buffer: Buffer,
    staging_buffer: Buffer,
    freelist: Vec<Range<usize>>,
    offsets: FnvHashMap<MeshId, Range<usize>>,
}

impl IndexBuffer {
    #[inline]
    fn new(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("index uberbuffer"),
            size: 1024 * 1024,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("vertex staging buffer"),
            size: 4 * 1024,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });
        Self {
            staging_buffer,
            freelist: vec![0..buffer.size() as usize],
            offsets: FnvHashMap::default(),
            buffer,
        }
    }
}

struct InstanceBuffer {
    buffer: Buffer,
    freelist: Vec<Range<usize>>,
    offsets: FnvHashMap<InstanceId, Range<usize>>,
}

impl InstanceBuffer {
    #[inline]
    fn new(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("instance uberbuffer"),
            size: 1024 * 1024,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            freelist: vec![0..buffer.size() as usize],
            offsets: FnvHashMap::default(),
            buffer,
        }
    }
}

struct IndirectBuffer {
    buffer: Buffer,
    freelist: Vec<Range<usize>>,
    offsets: FnvHashMap<InstanceId, Range<usize>>,
}

impl IndirectBuffer {
    #[inline]
    fn new(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("indirect uberbuffer"),
            size: 1024 * 1024,
            usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            freelist: vec![0..buffer.size() as usize],
            offsets: FnvHashMap::default(),
            buffer,
        }
    }
}

struct TextureBuffer {
    texture: WgpuTexture,
    staging_buffer: Buffer,
    freelist: Vec<Range<usize>>,
    offsets: FnvHashMap<TextureId, usize>,
}

impl TextureBuffer {
    #[inline]
    fn new(device: &Device) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("texture uberbuffer"),
            size: Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 256,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bc1RgbaUnormSrgb, // i.e. DXT1
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("texture staging buffer"),
            size: 1024 * 1024,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });
        Self {
            texture,
            staging_buffer,
            freelist: vec![0..256],
            offsets: FnvHashMap::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error(transparent)]
    InitializationError { source: anyhow::Error },
}

pub struct Wgpu {
    device: Device,
    queue: Queue,
    render_pipelines: FnvHashMap<ShaderId, RenderPipeline>,

    mesh_id: usize,
    instance_id: usize,
    texture_id: usize,

    surface: Surface,

    pending_upload_jobs: BinaryHeap<UploadJob>,
    current_upload_job: Option<UploadJob>,

    uploading_meshes: FnvHashMap<MeshId, Mesh>,
    complete_meshes: FnvHashSet<MeshId>,
    mesh_freelist: Vec<Mesh>,

    uploading_textures: FnvHashMap<TextureId, Texture>,
    complete_textures: FnvHashSet<TextureId>,
    texture_freelist: Vec<Texture>,

    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
    instance_buffer: InstanceBuffer,
    indirect_buffer: IndirectBuffer,
    texture_buffer: TextureBuffer,
}

impl Wgpu {
    pub fn new<W: HasRawWindowHandle + HasRawDisplayHandle>(
        window: &W,
        window_size: Vector2,
    ) -> Result<Self, WgpuError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..InstanceDescriptor::default()
        });
        let surface = unsafe { instance.create_surface(window) }.map_err(|e| {
            WgpuError::InitializationError {
                source: anyhow::anyhow!(e),
            }
        })?;

        let adapter = executor::block_on(instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..RequestAdapterOptions::default()
        }))
        .ok_or_else(|| WgpuError::InitializationError {
            source: anyhow::anyhow!("Failed to locate a suitable graphics adapter"),
        })?;

        let AdapterInfo { name, backend, .. } = adapter.get_info();
        log::debug!("Using adapter: \"{name}\" with {backend:?} backend");
        log::debug!("Adapter features: {:?}", adapter.features());
        log::debug!("Adapter limits: {:?}", adapter.limits());

        let (device, queue) = executor::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: Some("primary device"),
                features: Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | Features::TEXTURE_COMPRESSION_BC
                    | Features::TEXTURE_BINDING_ARRAY,
                limits: Limits::default(),
            },
            None,
        ))
        .map_err(|e| WgpuError::InitializationError {
            source: anyhow::anyhow!(e),
        })?;

        let surface_caps = surface.get_capabilities(&adapter);
        log::debug!("Surface caps: {:?}", surface_caps);

        let surface_format = surface_caps
            .formats
            .iter()
            .cloned()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);
        log::debug!("Surface format: {surface_format:?}");

        let size: (u32, u32) = window_size.into();
        surface.configure(
            &device,
            &SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.0,
                height: size.1,
                present_mode: surface_caps.present_modes[0],
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![],
            },
        );

        let vertex_buffer = VertexBuffer::new(&device);
        let index_buffer = IndexBuffer::new(&device);
        let instance_buffer = InstanceBuffer::new(&device);
        let indirect_buffer = IndirectBuffer::new(&device);
        let texture_buffer = TextureBuffer::new(&device);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let mut render_pipelines = FnvHashMap::default();
        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vertex_main",
                buffers: &[VertexBufferLayout {
                    array_stride: mem::size_of::<Vertex>() as BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x2,
                        3 => Float32x4,
                    ],
                }],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fragment_main",
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        render_pipelines.insert(ShaderId::Mesh, render_pipeline);

        Ok(Self {
            device,
            queue,
            render_pipelines,

            mesh_id: 0,
            instance_id: 0,
            texture_id: 0,

            surface,

            pending_upload_jobs: BinaryHeap::new(),
            current_upload_job: None,

            uploading_meshes: FnvHashMap::default(),
            complete_meshes: FnvHashSet::default(),
            mesh_freelist: Vec::new(),

            uploading_textures: FnvHashMap::default(),
            complete_textures: FnvHashSet::default(),
            texture_freelist: Vec::new(),

            vertex_buffer,
            index_buffer,
            instance_buffer,
            indirect_buffer,
            texture_buffer,
        })
    }

    pub fn pop_mesh(&mut self) -> Mesh {
        self.mesh_freelist.pop().unwrap_or_else(Mesh::new)
    }

    pub fn queue_mesh_upload(&mut self, mesh: Mesh, priority: usize) -> MeshId {
        // find smallest free space that fits
        let vertices_bytes_len = bytemuck::cast_slice::<_, u8>(mesh.vertices()).len();
        let (vertex_free_index, _) = self
            .vertex_buffer
            .freelist
            .iter()
            .enumerate()
            .find(|(_, range)| vertices_bytes_len <= range.len())
            .unwrap();

        let indices_bytes_len = bytemuck::cast_slice::<_, u8>(mesh.indices()).len();
        let (index_free_index, _) = self
            .index_buffer
            .freelist
            .iter()
            .enumerate()
            .find(|(_, range)| indices_bytes_len <= range.len())
            .unwrap();

        let vertex_range = self.vertex_buffer.freelist.remove(vertex_free_index);
        let index_range = self.index_buffer.freelist.remove(index_free_index);

        // re-insert any leftover free space
        if vertex_range.len() > vertices_bytes_len {
            let free_range = vertex_range.start + vertices_bytes_len..vertex_range.end;
            // insert sort by length and then by start position
            let index = self
                .vertex_buffer
                .freelist
                .binary_search_by(|other| match other.len().cmp(&free_range.len()) {
                    Ordering::Equal => other.start.cmp(&free_range.start),
                    ord => ord,
                })
                .unwrap_err(); // the `Err` is the insert position
            self.vertex_buffer.freelist.insert(index, free_range);
        }

        if index_range.len() > indices_bytes_len {
            let free_range = index_range.start + indices_bytes_len..index_range.end;
            // insert sort by length and then by start position
            let index = self
                .index_buffer
                .freelist
                .binary_search_by(|other| match other.len().cmp(&free_range.len()) {
                    Ordering::Equal => other.start.cmp(&free_range.start),
                    ord => ord,
                })
                .unwrap_err(); // the `Err` is the insert position
            self.index_buffer.freelist.insert(index, free_range);
        }

        let id = MeshId(self.mesh_id);
        self.mesh_id += 1;
        self.uploading_meshes.insert(id, mesh);
        self.pending_upload_jobs.push(UploadJob::Mesh {
            priority,
            id,
            src_vertex_offset: 0,
            src_index_offset: 0,
            dst_vertex_offset: vertex_range.start,
            dst_index_offset: index_range.start,
        });
        id
    }

    pub fn tick(&mut self) {
        if self.current_upload_job.is_none() {
            self.current_upload_job = self.pending_upload_jobs.pop();
        }
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("upload encoder"),
            });

        match self.current_upload_job {
            Some(UploadJob::Mesh {
                priority,
                id,
                src_vertex_offset,
                src_index_offset,
                dst_vertex_offset,
                dst_index_offset,
            }) => {
                let mesh = self.uploading_meshes.get(&id).unwrap();

                let vertices_bytes = &bytemuck::cast_slice(mesh.vertices());
                let vertices_bytes_remaining = vertices_bytes.len() - src_vertex_offset;
                let vertices_copy_size =
                    vertices_bytes_remaining.min(self.vertex_buffer.staging_buffer.size() as usize);
                let vertices_bytes =
                    &vertices_bytes[src_vertex_offset..(src_vertex_offset + vertices_copy_size)];

                {
                    let mut view = self
                        .vertex_buffer
                        .staging_buffer
                        .slice(0..vertices_copy_size as BufferAddress)
                        .get_mapped_range_mut();
                    view.copy_from_slice(vertices_bytes);
                }
                self.vertex_buffer.staging_buffer.unmap();

                encoder.copy_buffer_to_buffer(
                    &self.vertex_buffer.staging_buffer,
                    0,
                    &self.vertex_buffer.buffer,
                    dst_vertex_offset as BufferAddress,
                    vertices_copy_size as BufferAddress,
                );

                if vertices_copy_size == vertices_bytes_remaining {
                    self.current_upload_job = None;
                    self.complete_meshes.insert(id);
                } else {
                    self.current_upload_job = Some(UploadJob::Mesh {
                        priority,
                        id,
                        src_vertex_offset: src_vertex_offset + vertices_copy_size,
                        src_index_offset,
                        dst_vertex_offset: dst_index_offset + vertices_copy_size,
                        dst_index_offset,
                    });
                }
            }

            Some(_) => todo!(),
            None => {}
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn render_frame(&mut self) {
        let output = self.surface.get_current_texture().unwrap();
        let output_view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame render encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLUE),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(self.render_pipelines.get(&ShaderId::Mesh).unwrap());
            pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.buffer.slice(..), IndexFormat::Uint32);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
    }
}
