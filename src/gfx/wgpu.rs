use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    mem,
    ops::Range,
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc,
    },
};

use futures::executor;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{
    util::{align_to, BufferInitDescriptor, DeviceExt},
    AdapterInfo, AddressMode, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferAddress, BufferBindingType, BufferDescriptor, BufferUsages, BufferViewMut, Color,
    ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, CompareFunction,
    DepthBiasState, DepthStencilState, Device, DeviceDescriptor, Extent3d, Features, FilterMode,
    FragmentState, FrontFace, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, IndexFormat,
    Instance, InstanceDescriptor, Limits, LoadOp, MapMode, MultisampleState, Operations, Origin3d,
    PipelineLayoutDescriptor, PolygonMode, PresentMode, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType,
    SamplerDescriptor, ShaderStages, StencilState, Surface, SurfaceConfiguration,
    Texture as WgpuTexture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    VertexBufferLayout, VertexState, VertexStepMode,
};

use super::MipLevel;
use crate::{
    gfx::{Camera, Mesh, PerspectiveProjection, Texture, Vertex},
    math::{Matrix4, Quaternion, Vector2, Vector3},
    util::{FastHashMap, FastHashSet},
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

struct Chunk {
    buffer: Arc<Buffer>,
    used_offset: usize,
}

struct BufferStreamer {
    chunk_size: usize,
    chunk_name: &'static str,

    mapped_chunks: Vec<Chunk>,
    unmapped_chunks: Vec<Chunk>,
    free_chunks: Vec<Chunk>,

    tx: Sender<Chunk>,
    rx: Receiver<Chunk>,
}

impl BufferStreamer {
    fn new(chunk_size: usize, chunk_name: &'static str) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            chunk_size,
            chunk_name,
            mapped_chunks: Vec::new(),
            unmapped_chunks: Vec::new(),
            free_chunks: Vec::new(),
            tx,
            rx,
        }
    }

    fn stream_mip(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &WgpuTexture,
        layer: usize,
        mip: &MipLevel,
    ) -> BufferViewMut {
        let size = mip.data().len();
        let chunk = self.remove_mapped_chunk(device, size);

        let (width, height) = mip.size();
        encoder.copy_buffer_to_texture(
            ImageCopyBuffer {
                buffer: &chunk.buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(mip.bytes_per_row() as u32),
                    rows_per_image: Some(width as u32),
                },
            },
            ImageCopyTexture {
                texture: &target,
                mip_level: 0, // TODO: more mip levels. Probably stream all mips at once?
                origin: Origin3d {
                    x: 0,
                    y: 0,
                    z: layer as u32,
                },
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        self.insert_mapped_chunk(chunk, size)
    }

    fn stream_buffer(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        range: Range<usize>,
    ) -> BufferViewMut {
        let size = range.len();
        let chunk = self.remove_mapped_chunk(device, size);

        encoder.copy_buffer_to_buffer(
            &chunk.buffer,
            chunk.used_offset as BufferAddress,
            target,
            range.start as BufferAddress,
            size as BufferAddress,
        );

        self.insert_mapped_chunk(chunk, size)
    }

    fn unmap_all(&mut self) {
        for chunk in self.mapped_chunks.drain(..) {
            chunk.buffer.unmap();
            self.unmapped_chunks.push(chunk);
        }
    }

    fn remap_all(&mut self) {
        self.recv_chunks();
        let tx = &self.tx;
        for chunk in self.unmapped_chunks.drain(..) {
            let tx = tx.clone();
            chunk
                .buffer
                .clone()
                .slice(..)
                .map_async(MapMode::Write, move |_| {
                    let _ = tx.send(chunk);
                });
        }
    }

    fn remove_mapped_chunk(&mut self, device: &Device, size: usize) -> Chunk {
        if let Some(index) = self
            .mapped_chunks
            .iter()
            .position(|chunk| (chunk.used_offset + size) <= (chunk.buffer.size() as usize))
        {
            self.mapped_chunks.swap_remove(index)
        } else {
            self.recv_chunks(); // ensure self.free_chunks is up to date

            if let Some(index) = self
                .free_chunks
                .iter()
                .position(|chunk| size <= (chunk.buffer.size() as usize))
            {
                self.free_chunks.swap_remove(index)
            } else {
                Chunk {
                    buffer: Arc::new(device.create_buffer(&BufferDescriptor {
                        label: Some(self.chunk_name),
                        size: self.chunk_size as BufferAddress,
                        usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    })),
                    used_offset: 0,
                }
            }
        }
    }

    fn insert_mapped_chunk(&mut self, mut chunk: Chunk, size: usize) -> BufferViewMut {
        let old_offset = chunk.used_offset as BufferAddress;
        chunk.used_offset = align_to(chunk.used_offset + size, wgpu::MAP_ALIGNMENT as usize);

        self.mapped_chunks.push(chunk);
        self.mapped_chunks
            .last()
            .unwrap()
            .buffer
            .slice(old_offset..(old_offset + (size as BufferAddress)))
            .get_mapped_range_mut()
    }

    fn recv_chunks(&mut self) {
        while let Ok(mut chunk) = self.rx.try_recv() {
            chunk.used_offset = 0;
            self.free_chunks.push(chunk);
        }
    }
}

struct UberDataBuffer<I> {
    buffer: Buffer,
    streamer: BufferStreamer,
    freelist: Vec<Range<usize>>,
    offsets: FastHashMap<I, Range<usize>>,
}

#[derive(Debug)]
struct UberDataBufferDescriptor<'a> {
    device: &'a Device,
    name: &'static str,
    streamer_chunk_name: &'static str,
    size: usize,
    streamer_size: usize,
    usage: BufferUsages,
}

impl<I> UberDataBuffer<I> {
    #[inline]
    fn new(desc: &UberDataBufferDescriptor) -> Self {
        let buffer = desc.device.create_buffer(&BufferDescriptor {
            label: Some(desc.name),
            size: desc.size as BufferAddress,
            usage: desc.usage | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let streamer = BufferStreamer::new(desc.streamer_size, desc.streamer_chunk_name);
        Self {
            streamer,
            freelist: vec![0..buffer.size() as usize],
            offsets: FastHashMap::default(),
            buffer,
        }
    }
}

struct UberTextureBuffer {
    texture: WgpuTexture,
    streamer: BufferStreamer,
    freelist: Vec<Range<usize>>,
    offsets: FastHashMap<TextureId, usize>,
}

#[derive(Debug)]
struct UberTextureBufferDescriptor<'a> {
    device: &'a Device,
    name: &'static str,
    streamer_chunk_name: &'static str,
    width: usize,
    height: usize,
    layers: usize,
    format: TextureFormat,
}

impl UberTextureBuffer {
    #[inline]
    fn new(desc: &UberTextureBufferDescriptor) -> Self {
        let texture = desc.device.create_texture(&TextureDescriptor {
            label: Some(desc.name),
            size: Extent3d {
                width: desc.width as u32,
                height: desc.height as u32,
                depth_or_array_layers: desc.layers as u32,
            },
            mip_level_count: 1, // TODO: mips and sample count
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: desc.format,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let streamer = BufferStreamer::new(1024 * 1024, desc.streamer_chunk_name); // TODO: compute?
        Self {
            texture,
            streamer,
            freelist: vec![0..desc.layers],
            offsets: FastHashMap::default(),
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Default, Debug)]
struct Projection(Matrix4);

unsafe impl bytemuck::Zeroable for Projection {}
unsafe impl bytemuck::Pod for Projection {}

impl From<PerspectiveProjection> for Projection {
    #[inline]
    fn from(projection: PerspectiveProjection) -> Self {
        let projection_matrix = Matrix4::from(projection);
        Projection(&projection_matrix * &Matrix4::vulkan_projection_correct())
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Default, Debug)]
struct View {
    matrix: Matrix4,
    position: Vector3,
}

unsafe impl bytemuck::Zeroable for View {}
unsafe impl bytemuck::Pod for View {}

#[derive(Debug)]
struct ViewLookAt {
    view: View,
    at: Vector3,
}

impl From<Camera> for ViewLookAt {
    #[inline]
    fn from(camera: Camera) -> Self {
        let Camera {
            position,
            euler_angles,
        } = camera;
        let quat = Quaternion::from_angle_up(euler_angles.x())
            * Quaternion::from_angle_right(euler_angles.y());

        // Here we create a unit vector from the camera in the direction of the camera angle
        // I don't understand exactly why the rotation quaternion is "backward"
        let at = position - quat.forward_axis();

        // Then we can pass it to the handy look at matrix
        Self {
            view: View {
                matrix: Matrix4::look_at(position, at, Vector3::up()),
                position,
            },
            at,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error(transparent)]
    InitializationError { source: anyhow::Error },
}

#[derive(Debug)]
pub struct WgpuInitOptions {
    pub window_size: (usize, usize),
    pub projection: PerspectiveProjection,
    pub camera: Camera,
}

pub struct Wgpu {
    device: Device,
    queue: Queue,
    render_pipelines: FastHashMap<ShaderId, (BindGroup, RenderPipeline)>,

    mesh_id_counter: usize,
    instance_id_counter: usize,
    texture_id_counter: usize,

    surface: Surface,

    pending_upload_jobs: BinaryHeap<UploadJob>,
    current_upload_job: Option<UploadJob>,

    uploading_meshes: FastHashMap<MeshId, Mesh>,
    complete_meshes: FastHashSet<MeshId>,
    mesh_freelist: Vec<Mesh>,

    uploading_textures: FastHashMap<TextureId, Texture>,
    complete_textures: FastHashSet<TextureId>,
    texture_freelist: Vec<Texture>,

    projection: Projection,
    projection_buffer: Buffer,

    view_look_at: ViewLookAt,
    view_buffer: Buffer,

    vertex_buffer: UberDataBuffer<MeshId>,
    index_buffer: UberDataBuffer<MeshId>,
    instance_buffer: UberDataBuffer<InstanceId>,
    indirect_buffer: UberDataBuffer<InstanceId>,
    texture_buffer: UberTextureBuffer,
    depth_buffer: TextureView,
}

impl Wgpu {
    pub fn new<W: HasRawWindowHandle + HasRawDisplayHandle>(
        window: &W,
        opts: WgpuInitOptions,
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
        tracing::debug!("Using adapter: \"{name}\" with {backend:?} backend");
        tracing::debug!("Adapter features: {:?}", adapter.features());
        tracing::debug!("Adapter limits: {:?}", adapter.limits());

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
        tracing::debug!("Surface caps: {:?}", surface_caps);

        let surface_format = surface_caps
            .formats
            .iter()
            .cloned()
            .filter(TextureFormat::is_srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);
        tracing::debug!("Surface format: {surface_format:?}");

        let (width, height) = opts.window_size;
        surface.configure(
            &device,
            &SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: width as u32,
                height: height as u32,
                present_mode: PresentMode::Fifo,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![],
            },
        );

        let vertex_buffer = UberDataBuffer::new(&UberDataBufferDescriptor {
            device: &device,
            name: "vertex uberbuffer",
            streamer_chunk_name: "vertex stream chunk",
            size: 64 * 1024 * 1024,
            streamer_size: 1024 * 1024,
            usage: BufferUsages::VERTEX,
        });
        let index_buffer = UberDataBuffer::new(&UberDataBufferDescriptor {
            device: &device,
            name: "index uberbuffer",
            streamer_chunk_name: "index stream chunk",
            size: 32 * 1024 * 1024,
            streamer_size: 4 * 1024,
            usage: BufferUsages::INDEX,
        });
        let instance_buffer = UberDataBuffer::new(&UberDataBufferDescriptor {
            device: &device,
            name: "instance uberbuffer",
            streamer_chunk_name: "instance stream chunk",
            size: 32 * 1024,
            streamer_size: 0,
            usage: BufferUsages::STORAGE,
        });
        let indirect_buffer = UberDataBuffer::new(&UberDataBufferDescriptor {
            device: &device,
            name: "indirect uberbuffer",
            streamer_chunk_name: "indirect stream chunk",
            size: 1024 * 1024,
            streamer_size: 0,
            usage: BufferUsages::INDIRECT,
        });
        let texture_buffer = UberTextureBuffer::new(&UberTextureBufferDescriptor {
            device: &device,
            name: "texture uberbuffer",
            streamer_chunk_name: "texture stream chunk",
            width: 256,
            height: 256,
            layers: 256,
            format: TextureFormat::Bc3RgbaUnormSrgb,
        });

        let depth_buffer = device
            .create_texture(&TextureDescriptor {
                label: Some("depth buffer"),
                size: Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1, // TODO: mip levels
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor {
                aspect: TextureAspect::DepthOnly,
                ..Default::default()
            });

        let projection = Projection::from(opts.projection);
        let projection_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("projection buffer"),
            contents: bytemuck::bytes_of(&projection),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let view_look_at = ViewLookAt::from(opts.camera);
        let view_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("view buffer"),
            contents: bytemuck::bytes_of(&view_look_at.view),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("linear sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            compare: None,
            anisotropy_clamp: 1, // TODO: wat do?
            border_color: None,
        });
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("render pipeline bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0, // projection buffer
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None, // TODO: not optimal
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, // view buffer
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None, // TODO: not optimal
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2, // texture buffer
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3, // texture sampler
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4, // instance buffer
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: None, // TODO: not optimal
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("render pipeline bind group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0, // projection buffer
                    resource: projection_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1, // view buffer
                    resource: view_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2, // texture buffer
                    resource: BindingResource::TextureView(&texture_buffer.texture.create_view(
                        &TextureViewDescriptor {
                            label: Some("texture uberbuffer view"),
                            format: Some(TextureFormat::Bc3RgbaUnormSrgb),
                            dimension: Some(TextureViewDimension::D2Array),
                            aspect: TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: None,
                            base_array_layer: 0,
                            array_layer_count: None,
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 3, // texture sampler
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 4, // instance buffer
                    resource: instance_buffer.buffer.as_entire_binding(),
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
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
                        0 => Float32x3, // position
                        1 => Float32x3, // normal
                        2 => Float32x2, // tex_coord
                    ],
                }],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fragment_main",
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Cw,
                cull_mode: None, // TODO: culling
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let mut render_pipelines = FastHashMap::default();
        render_pipelines.insert(ShaderId::Mesh, (bind_group, render_pipeline));

        Ok(Self {
            device,
            queue,
            render_pipelines,

            mesh_id_counter: 0,
            instance_id_counter: 0,
            texture_id_counter: 0,

            surface,

            pending_upload_jobs: BinaryHeap::new(),
            current_upload_job: None,

            uploading_meshes: FastHashMap::default(),
            complete_meshes: FastHashSet::default(),
            mesh_freelist: Vec::new(),

            uploading_textures: FastHashMap::default(),
            complete_textures: FastHashSet::default(),
            texture_freelist: Vec::new(),

            projection,
            view_look_at,

            projection_buffer,
            view_buffer,

            vertex_buffer,
            index_buffer,
            instance_buffer,
            indirect_buffer,
            texture_buffer,
            depth_buffer,
        })
    }

    #[inline]
    pub fn set_projection(&mut self, projection: PerspectiveProjection) {
        self.projection = projection.into();
        self.queue.write_buffer(
            &self.projection_buffer,
            0,
            bytemuck::bytes_of(&self.projection),
        );
    }

    #[inline]
    pub fn set_camera(&mut self, camera: Camera) {
        self.view_look_at = camera.into();
        self.queue.write_buffer(
            &self.view_buffer,
            0,
            bytemuck::bytes_of(&self.view_look_at.view),
        );
    }

    #[inline]
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

        let id = MeshId(self.mesh_id_counter);
        self.mesh_id_counter += 1;
        self.uploading_meshes.insert(id, mesh);
        self.vertex_buffer.offsets.insert(id, vertex_range.clone());
        self.index_buffer.offsets.insert(id, index_range.clone());
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

    #[inline]
    pub fn pop_texture(&mut self) -> Texture {
        self.texture_freelist.pop().unwrap_or_else(Texture::new)
    }

    pub fn queue_texture_upload(&mut self, texture: Texture, priority: usize) -> TextureId {
        let texture_range = self.index_buffer.freelist.pop().unwrap();

        // re-insert any leftover free space
        if texture_range.len() > 1 {
            let free_range = texture_range.start + 1..texture_range.end;
            // insert sort by length and then by start position
            let index = self
                .texture_buffer
                .freelist
                .binary_search_by(|other| match other.len().cmp(&free_range.len()) {
                    Ordering::Equal => other.start.cmp(&free_range.start),
                    ord => ord,
                })
                .unwrap_err(); // the `Err` is the insert position
            self.texture_buffer.freelist.insert(index, free_range);
        }

        let id = TextureId(self.texture_id_counter);
        self.texture_id_counter += 1;
        self.uploading_textures.insert(id, texture);
        self.texture_buffer.offsets.insert(id, texture_range.start);
        self.pending_upload_jobs.push(UploadJob::Texture {
            priority,
            id,
            src_offset: 0,
            dst_layer: texture_range.start,
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
                label: Some("streaming upload encoder"),
            });

        match self.current_upload_job {
            None => {}

            Some(UploadJob::Mesh {
                priority,
                id,
                src_vertex_offset,
                src_index_offset,
                dst_vertex_offset,
                dst_index_offset,
            }) => {
                // TODO: factor into methods of the buffer structs
                let mesh = self.uploading_meshes.get(&id).unwrap();

                let vertices_bytes = &bytemuck::cast_slice(mesh.vertices());
                let vertices_bytes_remaining = vertices_bytes.len() - src_vertex_offset;
                let vertices_copy_size =
                    vertices_bytes_remaining.min(self.vertex_buffer.streamer.chunk_size);

                if vertices_copy_size != 0 {
                    tracing::debug!("Uploading {vertices_copy_size} vertex bytes for {id:?}");

                    let mut view = self.vertex_buffer.streamer.stream_buffer(
                        &self.device,
                        &mut encoder,
                        &self.vertex_buffer.buffer,
                        dst_vertex_offset..(dst_vertex_offset + vertices_copy_size),
                    );
                    view.copy_from_slice(
                        &vertices_bytes
                            [src_vertex_offset..(src_vertex_offset + vertices_copy_size)],
                    );
                }

                let indices_bytes = &bytemuck::cast_slice(mesh.indices());
                let indices_bytes_remaining = indices_bytes.len() - src_index_offset;
                let indices_copy_size =
                    indices_bytes_remaining.min(self.index_buffer.streamer.chunk_size);

                if indices_copy_size != 0 {
                    tracing::debug!("Uploading {indices_copy_size} index bytes for {id:?}");

                    let mut view = self.index_buffer.streamer.stream_buffer(
                        &self.device,
                        &mut encoder,
                        &self.index_buffer.buffer,
                        dst_index_offset..(dst_index_offset + indices_copy_size),
                    );
                    view.copy_from_slice(
                        &indices_bytes[src_index_offset..(src_index_offset + indices_copy_size)],
                    );
                }

                if indices_copy_size == indices_bytes_remaining
                    && vertices_copy_size == vertices_bytes_remaining
                {
                    tracing::debug!(
                        "Finished uploading {id:?}. ({} vertex bytes, {} index bytes)",
                        vertices_bytes.len(),
                        indices_bytes.len()
                    );
                    self.current_upload_job = None;
                    self.complete_meshes.insert(id);
                } else {
                    self.current_upload_job = Some(UploadJob::Mesh {
                        priority,
                        id,
                        src_vertex_offset: src_vertex_offset + vertices_copy_size,
                        src_index_offset: src_index_offset + indices_copy_size,
                        dst_vertex_offset: dst_index_offset + vertices_copy_size,
                        dst_index_offset: dst_index_offset + indices_copy_size,
                    });
                }
            }

            Some(UploadJob::Texture {
                priority,
                id,
                src_offset,
                dst_layer,
            }) => {
                let texture = self.uploading_textures.get(&id).unwrap();
                // get the first mip-level
                let mip = texture.mip_levels().next().unwrap();

                let texture_bytes_remaining = mip.data().len() - src_offset;
                let texture_copy_size =
                    texture_bytes_remaining.min(self.texture_buffer.streamer.chunk_size);

                if texture_copy_size != 0 {
                    tracing::debug!("Uploading {texture_copy_size} bytes for {id:?}");

                    // TODO: Right now we assume we have enough staging buffer to push
                    //   one whole layer at a time.
                    //   We'd need to compute a proper origin and extent for a given texel offset.
                    let mut view = self.texture_buffer.streamer.stream_mip(
                        &self.device,
                        &mut encoder,
                        &self.texture_buffer.texture,
                        dst_layer,
                        &mip,
                    );
                    view.copy_from_slice(&mip.data()[src_offset..(src_offset + texture_copy_size)]);
                }

                if texture_copy_size == texture_bytes_remaining {
                    tracing::debug!("Finished uploading {id:?}. ({} bytes)", mip.data().len());
                    self.current_upload_job = None;
                    self.complete_textures.insert(id);
                } else {
                    self.current_upload_job = Some(UploadJob::Texture {
                        priority,
                        id,
                        src_offset: src_offset + texture_copy_size,
                        dst_layer,
                    });
                }
            }
        }

        // must unmap before submtting. this will matter later when we do multiple jobs per tick
        self.vertex_buffer.streamer.unmap_all();
        self.index_buffer.streamer.unmap_all();
        self.texture_buffer.streamer.unmap_all();

        self.queue.submit(Some(encoder.finish()));

        // remap any chunks that are completed and ready to stream
        self.vertex_buffer.streamer.remap_all();
        self.index_buffer.streamer.remap_all();
        self.texture_buffer.streamer.remap_all();
    }

    pub fn render_frame(&mut self) {
        let output = self.surface.get_current_texture().unwrap();
        let output_texture_view = output
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
                    view: &output_texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLUE),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_buffer,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            let (bind_group, render_pipeline) = self.render_pipelines.get(&ShaderId::Mesh).unwrap();
            pass.set_bind_group(0, bind_group, &[0]);
            pass.set_pipeline(render_pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.buffer.slice(..), IndexFormat::Uint32);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
    }
}
