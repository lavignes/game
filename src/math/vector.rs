use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};
#[cfg(feature = "simd")]
use std::simd::{self, f32x2, f32x4, Simd, SimdFloat};

use crate::math::Quaternion;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Vector2(pub(crate) [f32; 2]);

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Vector3(pub(crate) [f32; 3]);

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Vector4(pub(crate) [f32; 4]);

macro_rules! vec_simd_binop {
    ($vec_type:ident, $simd_type:ident, $op_trait:ident, $op_name:ident) => {
        #[cfg(feature = "simd")]
        impl $op_trait<$vec_type> for $vec_type {
            type Output = $vec_type;

            #[inline]
            fn $op_name(self, rhs: $vec_type) -> Self::Output {
                let a = $simd_type::from_array(self.0);
                let b = $simd_type::from_array(rhs.0);
                $vec_type((a.$op_name(b)).to_array())
            }
        }

        #[cfg(feature = "simd")]
        impl $op_trait<$vec_type> for &$vec_type {
            type Output = $vec_type;

            #[inline]
            fn $op_name(self, rhs: $vec_type) -> Self::Output {
                let a = $simd_type::from_array(self.0);
                let b = $simd_type::from_array(rhs.0);
                $vec_type((a.$op_name(b)).to_array())
            }
        }

        #[cfg(feature = "simd")]
        impl $op_trait<&$vec_type> for $vec_type {
            type Output = $vec_type;

            #[inline]
            fn $op_name(self, rhs: &$vec_type) -> Self::Output {
                let a = $simd_type::from_array(self.0);
                let b = $simd_type::from_array(rhs.0);
                $vec_type((a.$op_name(b)).to_array())
            }
        }

        #[cfg(feature = "simd")]
        impl $op_trait<&$vec_type> for &$vec_type {
            type Output = $vec_type;

            #[inline]
            fn $op_name(self, rhs: &$vec_type) -> Self::Output {
                let a = $simd_type::from_array(self.0);
                let b = $simd_type::from_array(rhs.0);
                $vec_type((a.$op_name(b)).to_array())
            }
        }

        #[cfg(feature = "simd")]
        impl $op_trait<f32> for $vec_type {
            type Output = $vec_type;

            #[inline]
            fn $op_name(self, rhs: f32) -> Self::Output {
                let a = $simd_type::from_array(self.0);
                let b = Simd::splat(rhs);
                $vec_type((a.$op_name(b)).to_array())
            }
        }

        #[cfg(feature = "simd")]
        impl $op_trait<f32> for &$vec_type {
            type Output = $vec_type;

            #[inline]
            fn $op_name(self, rhs: f32) -> Self::Output {
                let a = $simd_type::from_array(self.0);
                let b = Simd::splat(rhs);
                $vec_type((a.$op_name(b)).to_array())
            }
        }
    };
}

macro_rules! vec_simd_misc {
    ($vec_type:ident, $simd_type:ident) => {
        #[cfg(feature = "simd")]
        impl Neg for $vec_type {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                Self((-$simd_type::from_array(self.0)).to_array())
            }
        }

        #[cfg(feature = "simd")]
        impl Neg for &$vec_type {
            type Output = $vec_type;

            #[inline]
            fn neg(self) -> Self::Output {
                $vec_type((-$simd_type::from_array(self.0)).to_array())
            }
        }

        #[cfg(feature = "simd")]
        impl PartialEq for $vec_type {
            #[inline]
            fn eq(&self, rhs: &Self) -> bool {
                let a = $simd_type::from_array(self.0);
                let b = $simd_type::from_array(rhs.0);
                // TODO: This isn't SIMD I think
                a == b
            }
        }

        #[cfg(feature = "simd")]
        impl Index<usize> for $vec_type {
            type Output = f32;

            #[inline]
            fn index(&self, index: usize) -> &f32 {
                &self.0[index]
            }
        }

        #[cfg(feature = "simd")]
        impl IndexMut<usize> for $vec_type {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut f32 {
                &mut self.0[index]
            }
        }
    };
}

macro_rules! vec2_binop {
    ($op_trait:ident, $op_name:ident) => {
        #[cfg(not(feature = "simd"))]
        impl $op_trait<Vector2> for Vector2 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: Vector2) -> Self::Output {
                Vector2([self.0[0].$op_name(rhs.0[0]), self.0[1].$op_name(rhs.0[1])])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<Vector2> for &Vector2 {
            type Output = Vector2;

            #[inline]
            fn $op_name(self, rhs: Vector2) -> Self::Output {
                Vector2([self.0[0].$op_name(rhs.0[0]), self.0[1].$op_name(rhs.0[1])])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<&Vector2> for Vector2 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: &Vector2) -> Self::Output {
                Vector2([self.0[0].$op_name(rhs.0[0]), self.0[1].$op_name(rhs.0[1])])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<&Vector2> for &Vector2 {
            type Output = Vector2;

            #[inline]
            fn $op_name(self, rhs: &Vector2) -> Self::Output {
                Vector2([self.0[0].$op_name(rhs.0[0]), self.0[1].$op_name(rhs.0[1])])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<f32> for Vector2 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: f32) -> Self::Output {
                Vector2([self.0[0].$op_name(rhs), self.0[1].$op_name(rhs)])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<f32> for &Vector2 {
            type Output = Vector2;

            #[inline]
            fn $op_name(self, rhs: f32) -> Self::Output {
                Vector2([self.0[0].$op_name(rhs), self.0[1].$op_name(rhs)])
            }
        }
    };
}

macro_rules! vec3_binop {
    ($op_trait:ident, $op_name:ident) => {
        impl $op_trait<Vector3> for Vector3 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: Vector3) -> Self::Output {
                Vector3([
                    self.0[0].$op_name(rhs.0[0]),
                    self.0[1].$op_name(rhs.0[1]),
                    self.0[2].$op_name(rhs.0[2]),
                ])
            }
        }

        impl $op_trait<Vector3> for &Vector3 {
            type Output = Vector3;

            #[inline]
            fn $op_name(self, rhs: Vector3) -> Self::Output {
                Vector3([
                    self.0[0].$op_name(rhs.0[0]),
                    self.0[1].$op_name(rhs.0[1]),
                    self.0[2].$op_name(rhs.0[2]),
                ])
            }
        }

        impl $op_trait<&Vector3> for Vector3 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: &Vector3) -> Self::Output {
                Vector3([
                    self.0[0].$op_name(rhs.0[0]),
                    self.0[1].$op_name(rhs.0[1]),
                    self.0[2].$op_name(rhs.0[2]),
                ])
            }
        }

        impl $op_trait<&Vector3> for &Vector3 {
            type Output = Vector3;

            #[inline]
            fn $op_name(self, rhs: &Vector3) -> Self::Output {
                Vector3([
                    self.0[0].$op_name(rhs.0[0]),
                    self.0[1].$op_name(rhs.0[1]),
                    self.0[2].$op_name(rhs.0[2]),
                ])
            }
        }

        impl $op_trait<f32> for Vector3 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: f32) -> Self::Output {
                Vector3([
                    self.0[0].$op_name(rhs),
                    self.0[1].$op_name(rhs),
                    self.0[2].$op_name(rhs),
                ])
            }
        }

        impl $op_trait<f32> for &Vector3 {
            type Output = Vector3;

            #[inline]
            fn $op_name(self, rhs: f32) -> Self::Output {
                Vector3([
                    self.0[0].$op_name(rhs),
                    self.0[1].$op_name(rhs),
                    self.0[2].$op_name(rhs),
                ])
            }
        }
    };
}

macro_rules! vec4_binop {
    ($op_trait:ident, $op_name:ident) => {
        #[cfg(not(feature = "simd"))]
        impl $op_trait<Vector4> for Vector4 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: Vector4) -> Self::Output {
                Vector4([
                    self.0[0].$op_name(rhs.0[0]),
                    self.0[1].$op_name(rhs.0[1]),
                    self.0[2].$op_name(rhs.0[2]),
                    self.0[3].$op_name(rhs.0[3]),
                ])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<Vector4> for &Vector4 {
            type Output = Vector4;

            #[inline]
            fn $op_name(self, rhs: Vector4) -> Self::Output {
                Vector4([
                    self.0[0].$op_name(rhs.0[0]),
                    self.0[1].$op_name(rhs.0[1]),
                    self.0[2].$op_name(rhs.0[2]),
                    self.0[3].$op_name(rhs.0[3]),
                ])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<&Vector4> for Vector4 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: &Vector4) -> Self::Output {
                Vector4([
                    self.0[0].$op_name(rhs.0[0]),
                    self.0[1].$op_name(rhs.0[1]),
                    self.0[2].$op_name(rhs.0[2]),
                    self.0[3].$op_name(rhs.0[3]),
                ])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<&Vector4> for &Vector4 {
            type Output = Vector4;

            #[inline]
            fn $op_name(self, rhs: &Vector4) -> Self::Output {
                Vector4([
                    self.0[0].$op_name(rhs.0[0]),
                    self.0[1].$op_name(rhs.0[1]),
                    self.0[2].$op_name(rhs.0[2]),
                    self.0[3].$op_name(rhs.0[3]),
                ])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<f32> for Vector4 {
            type Output = Self;

            #[inline]
            fn $op_name(self, rhs: f32) -> Self::Output {
                Vector4([
                    self.0[0].$op_name(rhs),
                    self.0[1].$op_name(rhs),
                    self.0[2].$op_name(rhs),
                    self.0[3].$op_name(rhs),
                ])
            }
        }

        #[cfg(not(feature = "simd"))]
        impl $op_trait<f32> for &Vector4 {
            type Output = Vector4;

            #[inline]
            fn $op_name(self, rhs: f32) -> Self::Output {
                Vector4([
                    self.0[0].$op_name(rhs),
                    self.0[1].$op_name(rhs),
                    self.0[2].$op_name(rhs),
                    self.0[3].$op_name(rhs),
                ])
            }
        }
    };
}

impl Vector2 {
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self([x, y])
    }

    #[inline]
    pub const fn splat(f: f32) -> Self {
        Self([f, f])
    }

    #[inline]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    #[inline]
    pub fn set_x(&mut self, f: f32) {
        self.0[0] = f;
    }

    #[inline]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

    #[inline]
    pub fn set_y(&mut self, f: f32) {
        self.0[1] = f;
    }

    #[inline]
    pub fn dot<Rhs: Into<Self>>(&self, rhs: Rhs) -> f32 {
        #[cfg(feature = "simd")]
        {
            let product = f32x2::from_array(self.0) * f32x2::from_array(rhs.into().0);
            product.reduce_sum()
        }

        #[cfg(not(feature = "simd"))]
        {
            let rhs = rhs.into();
            (self.0[0] * rhs.0[0]) + (self.0[1] * rhs.0[1])
        }
    }

    #[inline]
    pub fn normal_squared(&self) -> f32 {
        #[cfg(feature = "simd")]
        {
            let a = f32x2::from_array(self.0);
            (a * a).reduce_sum()
        }

        #[cfg(not(feature = "simd"))]
        {
            self.dot(*self)
        }
    }

    #[inline]
    pub fn length(&self) -> f32 {
        self.normal_squared().sqrt()
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        self / self.length()
    }

    #[inline]
    pub fn widened(&self, f: f32) -> Vector3 {
        Vector3([self.0[0], self.0[1], f])
    }
}

vec_simd_binop!(Vector2, f32x2, Add, add);
vec2_binop!(Add, add);
vec_simd_binop!(Vector2, f32x2, Sub, sub);
vec2_binop!(Sub, sub);
vec_simd_binop!(Vector2, f32x2, Mul, mul);
vec2_binop!(Mul, mul);
vec_simd_binop!(Vector2, f32x2, Div, div);
vec2_binop!(Div, div);
vec_simd_misc!(Vector2, f32x2);

#[cfg(not(feature = "simd"))]
impl PartialEq for Vector2 {
    #[inline]
    fn eq(&self, rhs: &Vector2) -> bool {
        (self.0[0] - rhs.0[0]).abs() <= f32::EPSILON && (self.0[1] - rhs.0[1]).abs() <= f32::EPSILON
    }
}

#[cfg(not(feature = "simd"))]
impl Neg for Vector2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self([-self.0[0], -self.0[1]])
    }
}

#[cfg(not(feature = "simd"))]
impl Neg for &Vector2 {
    type Output = Vector2;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector2([-self.0[0], -self.0[1]])
    }
}

#[cfg(not(feature = "simd"))]
impl Index<usize> for Vector2 {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

#[cfg(not(feature = "simd"))]
impl IndexMut<usize> for Vector2 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl From<(f32, f32)> for Vector2 {
    #[inline]
    fn from(value: (f32, f32)) -> Self {
        Self([value.0, value.1])
    }
}

impl From<(u32, u32)> for Vector2 {
    #[inline]
    fn from(value: (u32, u32)) -> Self {
        Self([value.0 as f32, value.1 as f32])
    }
}

impl From<(usize, usize)> for Vector2 {
    #[inline]
    fn from(value: (usize, usize)) -> Self {
        Self([value.0 as f32, value.1 as f32])
    }
}

impl Into<(u32, u32)> for Vector2 {
    #[inline]
    fn into(self) -> (u32, u32) {
        (self.0[0] as u32, self.0[1] as u32)
    }
}

impl Into<(f32, f32)> for Vector2 {
    #[inline]
    fn into(self) -> (f32, f32) {
        (self.0[0], self.0[1])
    }
}

impl Into<(usize, usize)> for Vector2 {
    #[inline]
    fn into(self) -> (usize, usize) {
        (self.0[0] as usize, self.0[1] as usize)
    }
}

impl Into<Vector3> for Vector2 {
    #[inline]
    fn into(self) -> Vector3 {
        self.widened(0.0)
    }
}

impl Into<Vector4> for Vector2 {
    #[inline]
    fn into(self) -> Vector4 {
        Vector4([self.0[0], self.0[1], 0.0, 0.0])
    }
}

impl Vector3 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self([x, y, z])
    }

    #[inline]
    pub const fn splat(f: f32) -> Self {
        Self([f, f, f])
    }

    #[inline]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    #[inline]
    pub fn set_x(&mut self, f: f32) {
        self.0[0] = f;
    }

    #[inline]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

    #[inline]
    pub fn set_y(&mut self, f: f32) {
        self.0[1] = f;
    }

    #[inline]
    pub const fn z(&self) -> f32 {
        self.0[2]
    }

    #[inline]
    pub fn set_z(&mut self, f: f32) {
        self.0[2] = f;
    }

    #[inline]
    pub fn length(&self) -> f32 {
        self.normal_squared().sqrt()
    }

    #[inline]
    pub fn normal_squared(&self) -> f32 {
        self.dot(*self)
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        self / self.length()
    }

    #[inline]
    pub fn cross<Rhs: Into<Self>>(&self, rhs: Rhs) -> Self {
        let rhs = rhs.into();
        Self([
            self.0[1] * rhs.0[2] - self.0[2] * rhs.0[1],
            self.0[2] * rhs.0[0] - self.0[0] * rhs.0[2],
            self.0[0] * rhs.0[1] - self.0[1] * rhs.0[0],
        ])
    }

    #[inline]
    pub fn dot<Rhs: Into<Self>>(&self, rhs: Rhs) -> f32 {
        let rhs = rhs.into();
        (self.0[0] * rhs.0[0]) + (self.0[1] * rhs.0[1]) + (self.0[2] * rhs.0[2])
    }

    #[inline]
    pub fn widened(&self, f: f32) -> Vector4 {
        Vector4([self.0[0], self.0[1], self.0[2], f])
    }

    #[inline]
    pub const fn up() -> Self {
        Self([0.0, 1.0, 0.0])
    }

    #[inline]
    pub const fn down() -> Self {
        Self([0.0, -1.0, 0.0])
    }

    #[inline]
    pub const fn right() -> Self {
        Self([1.0, 0.0, 0.0])
    }

    #[inline]
    pub const fn left() -> Self {
        Self([-1.0, 0.0, 0.0])
    }

    #[inline]
    pub const fn forward() -> Self {
        Self([0.0, 0.0, 1.0])
    }

    #[inline]
    pub const fn backward() -> Self {
        Self([0.0, 0.0, -1.0])
    }

    #[inline]
    pub fn rotated<T: Into<Quaternion>>(&self, rotation: T) -> Self {
        let rotation = rotation.into();
        (rotation * self * rotation.conjugated()).0.narrowed()
    }
}

vec3_binop!(Add, add);
vec3_binop!(Sub, sub);
vec3_binop!(Mul, mul);
vec3_binop!(Div, div);

impl PartialEq for Vector3 {
    #[inline]
    fn eq(&self, rhs: &Vector3) -> bool {
        (self.0[0] - rhs.0[0]).abs() <= f32::EPSILON
            && (self.0[1] - rhs.0[1]).abs() <= f32::EPSILON
            && (self.0[2] - rhs.0[2]).abs() <= f32::EPSILON
    }
}

impl Neg for Vector3 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

impl Index<usize> for Vector3 {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

impl IndexMut<usize> for Vector3 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl AsRef<[f32]> for Vector3 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

impl From<(f32, f32, f32)> for Vector3 {
    #[inline]
    fn from(value: (f32, f32, f32)) -> Self {
        Self([value.0, value.1, value.2])
    }
}

impl From<(usize, usize, usize)> for Vector3 {
    #[inline]
    fn from(value: (usize, usize, usize)) -> Self {
        Self([value.0 as f32, value.1 as f32, value.2 as f32])
    }
}

impl Into<Vector4> for Vector3 {
    #[inline]
    fn into(self) -> Vector4 {
        self.widened(0.0)
    }
}

impl Vector4 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self([x, y, z, w])
    }

    #[inline]
    pub const fn splat(f: f32) -> Self {
        Self([f, f, f, f])
    }

    #[inline]
    pub const fn x(&self) -> f32 {
        self.0[0]
    }

    #[inline]
    pub fn set_x(&mut self, f: f32) {
        self.0[0] = f;
    }

    #[inline]
    pub const fn y(&self) -> f32 {
        self.0[1]
    }

    #[inline]
    pub fn set_y(&mut self, f: f32) {
        self.0[1] = f;
    }

    #[inline]
    pub const fn z(&self) -> f32 {
        self.0[2]
    }

    #[inline]
    pub fn set_z(&mut self, f: f32) {
        self.0[2] = f;
    }

    #[inline]
    pub const fn w(&self) -> f32 {
        self.0[3]
    }

    #[inline]
    pub fn set_w(&mut self, f: f32) {
        self.0[3] = f;
    }

    #[inline]
    pub fn dot<Rhs: Into<Self>>(&self, rhs: Rhs) -> f32 {
        #[cfg(feature = "simd")]
        {
            let product = f32x4::from_array(self.0) * f32x4::from_array(rhs.into().0);
            product.reduce_sum()
        }

        #[cfg(not(feature = "simd"))]
        {
            let rhs = rhs.into();
            (self.0[0] * rhs.0[0])
                + (self.0[1] * rhs.0[1])
                + (self.0[2] * rhs.0[2])
                + (self.0[3] * rhs.0[3])
        }
    }

    #[inline]
    pub fn cross<Rhs: Into<Self>>(&self, rhs: Rhs) -> Self {
        #[cfg(feature = "simd")]
        {
            let rhs = f32x4::from_array(rhs.into().0);
            let a = simd::simd_swizzle!(f32x4::from_array(self.0), [3, 0, 2, 1]);
            let b = simd::simd_swizzle!(rhs, [3, 1, 0, 2]);
            let c = a * rhs;
            let d = a * b;
            let e = simd::simd_swizzle!(c, [3, 0, 2, 1]);
            Self((d - e).to_array())
        }

        #[cfg(not(feature = "simd"))]
        {
            let rhs = rhs.into();
            self.narrowed().cross(rhs.narrowed()).widened(0.0)
        }
    }

    #[inline]
    pub fn normal_squared(&self) -> f32 {
        self.dot(*self)
    }

    #[inline]
    pub fn length(&self) -> f32 {
        self.normal_squared().sqrt()
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        self / self.length()
    }

    #[inline]
    pub fn narrowed(&self) -> Vector3 {
        Vector3([self.0[0], self.0[1], self.0[2]])
    }
}

vec_simd_binop!(Vector4, f32x4, Add, add);
vec4_binop!(Add, add);
vec_simd_binop!(Vector4, f32x4, Sub, sub);
vec4_binop!(Sub, sub);
vec_simd_binop!(Vector4, f32x4, Mul, mul);
vec4_binop!(Mul, mul);
vec_simd_binop!(Vector4, f32x4, Div, div);
vec4_binop!(Div, div);
vec_simd_misc!(Vector4, f32x4);

#[cfg(not(feature = "simd"))]
impl PartialEq for Vector4 {
    #[inline]
    fn eq(&self, rhs: &Vector4) -> bool {
        (self.0[0] - rhs.0[0]).abs() <= f32::EPSILON
            && (self.0[1] - rhs.0[1]).abs() <= f32::EPSILON
            && (self.0[2] - rhs.0[2]).abs() <= f32::EPSILON
            && (self.0[3] - rhs.0[3]).abs() <= f32::EPSILON
    }
}

#[cfg(not(feature = "simd"))]
impl Neg for Vector4 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

#[cfg(not(feature = "simd"))]
impl Neg for &Vector4 {
    type Output = Vector4;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector4([-self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }
}

#[cfg(not(feature = "simd"))]
impl Index<usize> for Vector4 {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}

#[cfg(not(feature = "simd"))]
impl IndexMut<usize> for Vector4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}
