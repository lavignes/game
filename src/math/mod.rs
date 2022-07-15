use bytemuck::{Pod, Zeroable};

mod matrix;
mod quaternion;
mod vector;

pub use matrix::*;
pub use quaternion::*;
pub use vector::*;

unsafe impl Zeroable for Vector2 {}
unsafe impl Pod for Vector2 {}

unsafe impl Zeroable for Vector3 {}
unsafe impl Pod for Vector3 {}

unsafe impl Zeroable for Vector4 {}
unsafe impl Pod for Vector4 {}

unsafe impl Zeroable for Matrix3 {}
unsafe impl Pod for Matrix3 {}

unsafe impl Zeroable for Matrix4 {}
unsafe impl Pod for Matrix4 {}

unsafe impl Zeroable for Quaternion {}
unsafe impl Pod for Quaternion {}
