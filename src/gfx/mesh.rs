use bytemuck::{Pod, Zeroable};

use crate::math::{Vector2, Vector3, Vector4};

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Vertex {
    pub position: Vector3,
    pub normal: Vector3,
    pub tex_coord: Vector2,
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    #[inline]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    #[inline]
    pub fn push_vertices(&mut self, vertices: &[Vertex]) {
        self.vertices.extend_from_slice(vertices)
    }

    #[inline]
    pub fn push_indices(&mut self, indices: &[u32]) {
        self.indices.extend_from_slice(indices)
    }

    #[inline]
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    #[inline]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
}
