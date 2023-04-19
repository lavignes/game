use std::{cmp::Ordering, collections::BinaryHeap, mem, num::NonZeroU32, ops::Range};

use fnv::{FnvHashMap, FnvHashSet};
use futures::executor;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    AdapterInfo, AddressMode, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferAddress, BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, ColorWrites, CommandEncoderDescriptor, CompareFunction, DepthBiasState,
    DepthStencilState, Device, DeviceDescriptor, Extent3d, Features, FilterMode, FragmentState,
    FrontFace, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, IndexFormat, Instance,
    InstanceDescriptor, Limits, LoadOp, MultisampleState, Operations, Origin3d,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType,
    SamplerDescriptor, ShaderStages, StencilState, Surface, SurfaceConfiguration,
    Texture as WgpuTexture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension,
    VertexBufferLayout, VertexState, VertexStepMode,
};

use crate::{
    gfx::{Camera, Mesh, PerspectiveProjection, Texture, Vertex},
    math::{Matrix4, Quaternion, Vector2, Vector3},
};

const TEXTURE_WIDTH: u32 = 256;
const TEXTURE_HEIGHT: u32 = 256;
const TEXTURE_LAYERS: u32 = 256;

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
            size: 64 * 1024 * 1024,
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
            size: 32 * 1024 * 1024,
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
            size: 32 * 1024,
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
                width: TEXTURE_WIDTH,
                height: TEXTURE_HEIGHT,
                depth_or_array_layers: TEXTURE_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bc1RgbaUnormSrgb, // i.e. DXT1 SRGB
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
            freelist: vec![0..TEXTURE_LAYERS as usize],
            offsets: FnvHashMap::default(),
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
    pub window_size: Vector2,
    pub projection: PerspectiveProjection,
    pub camera: Camera,
}

pub struct Wgpu {
    device: Device,
    queue: Queue,
    render_pipelines: FnvHashMap<ShaderId, (BindGroup, RenderPipeline)>,

    mesh_id_counter: usize,
    instance_id_counter: usize,
    texture_id_counter: usize,

    surface: Surface,

    pending_upload_jobs: BinaryHeap<UploadJob>,
    current_upload_job: Option<UploadJob>,

    uploading_meshes: FnvHashMap<MeshId, Mesh>,
    complete_meshes: FnvHashSet<MeshId>,
    mesh_freelist: Vec<Mesh>,

    uploading_textures: FnvHashMap<TextureId, Texture>,
    complete_textures: FnvHashSet<TextureId>,
    texture_freelist: Vec<Texture>,

    projection: Projection,
    projection_buffer: Buffer,

    view_look_at: ViewLookAt,
    view_buffer: Buffer,

    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
    instance_buffer: InstanceBuffer,
    indirect_buffer: IndirectBuffer,
    texture_buffer: TextureBuffer,
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

        let size: (u32, u32) = opts.window_size.into();
        surface.configure(
            &device,
            &SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.0,
                height: size.1,
                present_mode: surface_caps.present_modes[0],
                //present_mode: wgpu::PresentMode::Immediate,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![],
            },
        );

        let vertex_buffer = VertexBuffer::new(&device);
        let index_buffer = IndexBuffer::new(&device);
        let instance_buffer = InstanceBuffer::new(&device);
        let indirect_buffer = IndirectBuffer::new(&device);
        let texture_buffer = TextureBuffer::new(&device);
        let depth_buffer = device
            .create_texture(&TextureDescriptor {
                label: Some("depth buffer"),
                size: Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
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
            label: Some("basic sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            compare: None,
            anisotropy_clamp: None,
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
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1, // view buffer
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2, // texture buffer
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3, // texture sampler
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering {}),
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
                            format: Some(TextureFormat::Bc1RgbaUnormSrgb),
                            dimension: Some(TextureViewDimension::D2Array),
                            aspect: TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: NonZeroU32::new(1),
                            base_array_layer: 0,
                            array_layer_count: NonZeroU32::new(TEXTURE_LAYERS),
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
                        0 => Float32x3,
                        1 => Float32x3,
                        2 => Float32x2,
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

        let mut render_pipelines = FnvHashMap::default();
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

            uploading_meshes: FnvHashMap::default(),
            complete_meshes: FnvHashSet::default(),
            mesh_freelist: Vec::new(),

            uploading_textures: FnvHashMap::default(),
            complete_textures: FnvHashSet::default(),
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
                let mesh = self.uploading_meshes.get(&id).unwrap();

                let vertices_bytes = &bytemuck::cast_slice(mesh.vertices());
                let vertices_bytes_remaining = vertices_bytes.len() - src_vertex_offset;
                let vertices_copy_size =
                    vertices_bytes_remaining.min(self.vertex_buffer.staging_buffer.size() as usize);

                if vertices_copy_size != 0 {
                    log::debug!("Uploading {vertices_copy_size} vertex bytes for {id:?}");
                    let vertices_bytes = &vertices_bytes
                        [src_vertex_offset..(src_vertex_offset + vertices_copy_size)];
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
                }

                let indices_bytes = &bytemuck::cast_slice(mesh.indices());
                let indices_bytes_remaining = indices_bytes.len() - src_index_offset;
                let indices_copy_size =
                    indices_bytes_remaining.min(self.index_buffer.staging_buffer.size() as usize);

                if indices_copy_size != 0 {
                    log::debug!("Uploading {indices_copy_size} index bytes for {id:?}");
                    let indices_bytes =
                        &indices_bytes[src_index_offset..(src_index_offset + indices_copy_size)];
                    {
                        let mut view = self
                            .index_buffer
                            .staging_buffer
                            .slice(0..indices_copy_size as BufferAddress)
                            .get_mapped_range_mut();
                        view.copy_from_slice(indices_bytes);
                    }
                    self.index_buffer.staging_buffer.unmap();

                    encoder.copy_buffer_to_buffer(
                        &self.index_buffer.staging_buffer,
                        0,
                        &self.index_buffer.buffer,
                        dst_index_offset as BufferAddress,
                        indices_copy_size as BufferAddress,
                    );
                }

                if indices_copy_size == indices_bytes_remaining
                    && vertices_copy_size == vertices_bytes_remaining
                {
                    log::debug!(
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
                    texture_bytes_remaining.min(self.texture_buffer.staging_buffer.size() as usize);

                if texture_copy_size != 0 {
                    log::debug!("Uploading {texture_copy_size} bytes for {id:?}");
                    let texture_bytes = &mip.data()[src_offset..(src_offset + texture_copy_size)];
                    {
                        let mut view = self
                            .texture_buffer
                            .staging_buffer
                            .slice(0..texture_copy_size as BufferAddress)
                            .get_mapped_range_mut();
                        view.copy_from_slice(texture_bytes);
                    }
                    self.texture_buffer.staging_buffer.unmap();

                    // TODO: Right now we assume we have enough staging buffer to push
                    //   one whole layer at a time.
                    //   We'd need to compute a proper origin and extent for a given texel offset.
                    encoder.copy_buffer_to_texture(
                        ImageCopyBuffer {
                            buffer: &self.texture_buffer.staging_buffer,
                            layout: ImageDataLayout {
                                offset: 0,
                                bytes_per_row: NonZeroU32::new(mip.bytes_per_row() as u32),
                                rows_per_image: NonZeroU32::new(TEXTURE_HEIGHT),
                            },
                        },
                        ImageCopyTexture {
                            texture: &self.texture_buffer.texture,
                            mip_level: 0,
                            origin: Origin3d {
                                x: 0,
                                y: 0,
                                z: dst_layer as u32,
                            },
                            aspect: TextureAspect::All,
                        },
                        Extent3d {
                            width: TEXTURE_WIDTH,
                            height: TEXTURE_HEIGHT,
                            depth_or_array_layers: 1,
                        },
                    );
                }

                if texture_copy_size == texture_bytes_remaining {
                    log::debug!("Finished uploading {id:?}. ({} bytes)", mip.data().len());
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

        self.queue.submit(Some(encoder.finish()));
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
