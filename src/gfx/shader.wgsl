struct Projection {
    matrix: mat4x4<f32>,
}

struct View {
    matrix: mat4x4<f32>,
    position: vec3<f32>,
}

@group(0) @binding(0) var<uniform> projection: Projection;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var textures: texture_2d_array<f32>;
@group(0) @binding(3) var texture_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coord: vec2<f32>,
}

@vertex
fn vertex_main(in: VertexInput) -> VertexOutput {
    let vm = mat4x4<f32>(
        vec4<f32>(0.2, -0.2, 0.0, 0.0),
        vec4<f32>(0.2, 0.2, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    // TODO: model * vec4<f32>(in.position, 1.0);
    let world_position = vec4<f32>(in.position, 1.0);
    //let clip_position = projection.matrix * view.matrix * world_position;
    let clip_position = view.matrix * world_position;

    var out: VertexOutput;
    out.clip_position = clip_position;
    out.world_position = world_position.xyz;
    out.tex_coord = in.tex_coord;
    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(textures, texture_sampler, in.tex_coord, 0);
    if color.a == 0.0 {
        discard;
    }
    return color;
}

