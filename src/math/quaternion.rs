use std::{
    f32::consts,
    ops::{Mul, MulAssign},
    simd::{self, f32x4, SimdFloat},
};

use crate::math::{Vector3, Vector4};

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Quaternion(pub Vector4);

impl Quaternion {
    #[inline]
    pub const fn identity() -> Self {
        Self(Vector4([0.0, 0.0, 0.0, 1.0]))
    }

    #[inline]
    pub fn from_axis_angle<Axis: Into<Vector3>>(axis: Axis, angle: f32) -> Self {
        let half_theta = angle / 2.0;
        let sin_half_theta = half_theta.sin();
        let cos_half_theta = half_theta.cos();
        Self((axis.into() * sin_half_theta).widened(cos_half_theta))
    }

    #[inline]
    pub fn from_angle_right(angle: f32) -> Self {
        Self::from_axis_angle(Vector3::right(), angle)
    }

    #[inline]
    pub fn from_angle_up(angle: f32) -> Self {
        Self::from_axis_angle(Vector3::up(), angle)
    }

    #[inline]
    pub fn from_angle_forward(angle: f32) -> Self {
        Self::from_axis_angle(Vector3::forward(), angle)
    }

    pub fn look_at<Pos, At, Up>(position: Pos, at: At, up: Up) -> Self
    where
        Pos: Into<Vector3>,
        At: Into<Vector3>,
        Up: Into<Vector3>,
    {
        let forward = (at.into() - position.into()).normalized();
        let dot = Vector3::forward().dot(forward);
        if (dot + 1.0).abs() <= f32::EPSILON {
            return Self::from_axis_angle(up.into(), consts::PI);
        }
        if (dot - 1.0).abs() <= f32::EPSILON {
            return Self::identity();
        }
        let angle = dot.acos();
        let axis = Vector3::forward().cross(forward).normalized();
        Self::from_axis_angle(axis, angle)
    }

    #[inline]
    pub fn normalized(&self) -> Self {
        Self(self.0.normalized())
    }

    #[inline]
    pub fn conjugated(&self) -> Self {
        let a = f32x4::from_array(self.0 .0);
        let b = f32x4::from_array([-1.0, -1.0, -1.0, 1.0]);
        Self(Vector4((a * b).to_array()))
    }

    #[inline]
    pub fn right_axis(&self) -> Vector3 {
        Vector3::right().rotated(*self)
    }

    #[inline]
    pub fn left_axis(&self) -> Vector3 {
        Vector3::left().rotated(*self)
    }

    #[inline]
    pub fn up_axis(&self) -> Vector3 {
        Vector3::up().rotated(*self)
    }

    #[inline]
    pub fn forward_axis(&self) -> Vector3 {
        Vector3::forward().rotated(*self)
    }

    /// A *very* fast interpolation. Only really useful for
    /// interpolating as long as the quaternions are aligned with
    /// an axis.
    pub fn lerp<Rhs: Into<Quaternion>>(&self, rhs: Rhs, dt: f32) -> Self {
        let rhs = rhs.into();
        let cos_half_theta = self.0.dot(rhs.0);
        if cos_half_theta < 0.0 {
            return Self(((-self.0) - rhs.0) * dt + self.0);
        }
        Self((self.0 - rhs.0) * dt + self.0)
    }

    /// A fast interpolation. Only really useful for
    /// interpolating as long as the quaternions are aligned with
    /// an axis.
    #[inline]
    pub fn nlerp<Rhs: Into<Quaternion>>(&self, rhs: Rhs, dt: f32) -> Self {
        self.lerp(rhs, dt).normalized()
    }

    /// Interpolate between two quaternions.
    #[inline]
    pub fn slerp<Rhs: Into<Quaternion>>(&self, rhs: Rhs, dt: f32) -> Self {
        let rhs = rhs.into();
        let cos_half_theta = self.0.dot(rhs.0);
        if cos_half_theta.abs() >= 1.0 {
            return *self;
        }
        let sin_half_theta = (1.0 - cos_half_theta * cos_half_theta).sqrt();
        if sin_half_theta.abs() <= f32::EPSILON {
            return Self(self.0 * 0.5 + rhs.0 * 0.5);
        }
        let half_theta = cos_half_theta.acos();
        let a = ((1.0 - dt) * half_theta).sin() / sin_half_theta;
        let b = (dt * half_theta).sin() / sin_half_theta;
        Self(self.0 * a + rhs.0 * b)
    }
}

impl MulAssign<Quaternion> for Quaternion {
    #[inline]
    fn mul_assign(&mut self, rhs: Quaternion) {
        *self = *self * rhs;
    }
}

impl MulAssign<&Quaternion> for Quaternion {
    #[inline]
    fn mul_assign(&mut self, rhs: &Quaternion) {
        *self = *self * rhs;
    }
}

#[rustfmt::skip]
macro_rules! quat_mul {
    ($self_type:ty, $quat_type:ty) => {
        impl Mul<$quat_type> for $self_type {
            type Output = Quaternion;

            fn mul(self, rhs: $quat_type) -> Quaternion {
                let a = f32x4::from_array(self.0.0);
                let b = f32x4::from_array(rhs.0.0);

                // TODO(lavignes): could replace these with bitwise ops to flip the sign
                let sig_mask1 = f32x4::from_array([1.0, 1.0, 1.0, -1.0]);
                let sig_mask2 = f32x4::from_array([1.0, -1.0, -1.0, -1.0]);

                let a0312 = simd::simd_swizzle!(a, [0, 3, 1, 2]);
                let b3021 = simd::simd_swizzle!(b, [3, 0, 2, 1]);
                let s = (a0312 * b3021 * sig_mask1).reduce_sum();

                let a1320 = simd::simd_swizzle!(a, [1, 3, 2, 0]);
                let b3102 = simd::simd_swizzle!(b, [3, 1, 0, 2]);
                let t = (a1320 * b3102 * sig_mask1).reduce_sum();

                let a2301 = simd::simd_swizzle!(a, [2, 3, 0, 1]);
                let b3210 = simd::simd_swizzle!(b, [3, 2, 1, 0]);
                let u = (a2301 * b3210 * sig_mask1).reduce_sum();

                let a3012 = simd::simd_swizzle!(a, [3, 0, 1, 2]);
                let b3012 = simd::simd_swizzle!(b, [3, 0, 1, 2]);
                let v = (a3012 * b3012 * sig_mask2).reduce_sum();

                Quaternion(Vector4([s, t, u, v]))
            }
        }
    };
}

quat_mul!(Quaternion, Quaternion);
quat_mul!(&Quaternion, Quaternion);
quat_mul!(Quaternion, &Quaternion);
quat_mul!(&Quaternion, &Quaternion);

macro_rules! quat_vec3_mul {
    ($self_type:ty, $vec_type:ty) => {
        impl Mul<$vec_type> for $self_type {
            type Output = Quaternion;

            #[inline]
            fn mul(self, rhs: $vec_type) -> Quaternion {
                // AFAICT converting the vec into a quat works!
                self * &Quaternion(rhs.widened(0.0))
            }
        }
    };
}

quat_vec3_mul!(Quaternion, Vector3);
quat_vec3_mul!(&Quaternion, Vector3);
quat_vec3_mul!(Quaternion, &Vector3);
quat_vec3_mul!(&Quaternion, &Vector3);
