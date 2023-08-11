use std::ops::Mul;
#[cfg(feature = "simd")]
use std::simd::{self, f32x4, SimdFloat, SimdPartialOrd, Which};

use crate::math::{Quaternion, Vector3, Vector4};

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Matrix3(pub [Vector3; 3]);

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Matrix4(pub [Vector4; 4]);

impl Matrix4 {
    #[inline]
    pub const fn new(x: Vector4, y: Vector4, z: Vector4, w: Vector4) -> Self {
        Self([x, y, z, w])
    }

    #[inline]
    pub const fn identity() -> Self {
        Self([
            Vector4([1.0, 0.0, 0.0, 0.0]),
            Vector4([0.0, 1.0, 0.0, 0.0]),
            Vector4([0.0, 0.0, 1.0, 0.0]),
            Vector4([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    #[inline]
    pub const fn translate(v: Vector3) -> Self {
        Self([
            Vector4([1.0, 0.0, 0.0, 0.0]),
            Vector4([0.0, 1.0, 0.0, 0.0]),
            Vector4([0.0, 0.0, 1.0, 0.0]),
            Vector4([v.0[0], v.0[1], v.0[2], 1.0]),
        ])
    }

    #[inline]
    pub const fn scale(v: Vector3) -> Self {
        Self([
            Vector4([v.0[0], 0.0, 0.0, 0.0]),
            Vector4([0.0, v.0[1], 0.0, 0.0]),
            Vector4([0.0, 0.0, v.0[2], 0.0]),
            Vector4([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    #[inline]
    pub fn rotate_right(angle: f32) -> Self {
        let sin_theta = angle.sin();
        let cos_theta = angle.cos();
        Self([
            Vector4([1.0, 0.0, 0.0, 0.0]),
            Vector4([0.0, cos_theta, -sin_theta, 0.0]),
            Vector4([0.0, sin_theta, cos_theta, 0.0]),
            Vector4([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    #[inline]
    pub fn rotate_up(angle: f32) -> Self {
        let sin_theta = angle.sin();
        let cos_theta = angle.cos();
        Self([
            Vector4([cos_theta, 0.0, sin_theta, 0.0]),
            Vector4([0.0, 1.0, 0.0, 0.0]),
            Vector4([-sin_theta, 0.0, cos_theta, 0.0]),
            Vector4([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    #[inline]
    pub fn rotate_forward(angle: f32) -> Self {
        let sin_theta = angle.sin();
        let cos_theta = angle.cos();
        Self([
            Vector4([cos_theta, -sin_theta, 0.0, 0.0]),
            Vector4([sin_theta, cos_theta, 0.0, 0.0]),
            Vector4([0.0, 0.0, 1.0, 0.0]),
            Vector4([0.0, 0.0, 0.0, 1.0]),
        ])
    }

    #[inline]
    pub fn perspective(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        let depth = near - far;
        let tan_fov = (fov / 2.0).tan();
        Self([
            Vector4([1.0 / (tan_fov * aspect_ratio), 0.0, 0.0, 0.0]),
            Vector4([0.0, 1.0 / tan_fov, 0.0, 0.0]),
            Vector4([0.0, 0.0, (near + far) / depth, -1.0]),
            Vector4([0.0, 0.0, (2.0 * far * near) / depth, 0.0]),
        ])
    }

    #[inline]
    pub fn orthographic(top: f32, left: f32, bottom: f32, right: f32, near: f32, far: f32) -> Self {
        #[cfg(feature = "simd")]
        {
            // We make `b` end in 1.0, so the final vec ends in 1.0 after the negation
            let a = f32x4::from_array([right, top, far, 0.0]);
            let b = f32x4::from_array([right, top, far, 1.0]);
            let c = a + b;
            let d = a - b;
            Self([
                Vector4([2.0 / (right - left), 0.0, 0.0, 0.0]),
                Vector4([0.0, 2.0 / (top - bottom), 0.0, 0.0]),
                Vector4([0.0, 0.0, -2.0 / (far - near), 0.0]),
                Vector4((-(c / d)).to_array()),
            ])
        }

        #[cfg(not(feature = "simd"))]
        {
            Self([
                Vector4([2.0 / (right - left), 0.0, 0.0, 0.0]),
                Vector4([0.0, 2.0 / (top - bottom), 0.0, 0.0]),
                Vector4([0.0, 0.0, -2.0 / (far - near), 0.0]),
                Vector4([
                    -((right + left) / (right - left)),
                    -((top + bottom) / (top - bottom)),
                    -((far + near) / (far - near)),
                    1.0,
                ]),
            ])
        }
    }

    #[inline]
    pub const fn vulkan_projection_correct() -> Self {
        Self([
            Vector4([-1.0, 0.0, 0.0, 0.0]),
            Vector4([0.0, 1.0, 0.0, 0.0]),
            Vector4([0.0, 0.0, 0.5, 0.0]),
            Vector4([0.0, 0.0, 0.5, 1.0]),
        ])
    }

    #[inline]
    pub fn look_at<Pos, At, Up>(position: Pos, at: At, up: Up) -> Self
    where
        Pos: Into<Vector4>,
        At: Into<Vector4>,
        Up: Into<Vector4>,
    {
        let position = position.into();

        #[cfg(feature = "simd")]
        {
            let z = (at.into() - position).normalized();
            let x = z.cross(up.into()).normalized();
            let y = x.cross(z);
            let z = -z;
            let w = f32x4::from_array([x.dot(position), y.dot(position), z.dot(position), -1.0]);
            Self([
                Vector4([x.0[0], y.0[0], z.0[0], 0.0]),
                Vector4([x.0[1], y.0[1], z.0[1], 0.0]),
                Vector4([x.0[2], y.0[2], z.0[2], 0.0]),
                Vector4((-w).to_array()),
            ])
        }

        #[cfg(not(feature = "simd"))]
        {
            let z = (at.into() - position).normalized();
            let x = z.cross(up).normalized();
            let y = x.cross(z);
            let z = -z;
            Self([
                Vector4([x.0[0], y.0[0], z.0[0], 0.0]),
                Vector4([x.0[1], y.0[1], z.0[1], 0.0]),
                Vector4([x.0[2], y.0[2], z.0[2], 0.0]),
                Vector4([-x.dot(position), -y.dot(position), -z.dot(position), 1.0]),
            ])
        }
    }

    /// A fast transform matrix inversion that only works if the matrix has no scale components.
    // https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
    #[rustfmt::skip]
    #[cfg(feature = "simd")]
    pub fn inversed_transform_no_scale(&self) -> Self {
        let x = f32x4::from_array(self.0[0].0);
        let y = f32x4::from_array(self.0[1].0);
        let z = f32x4::from_array(self.0[2].0);

        // transpose the 3x3 sub-matrix
        let temp0 = simd::simd_swizzle!(x, y, [Which::First(0), Which::First(1), Which::Second(0), Which::Second(1)]);
        let temp1 = simd::simd_swizzle!(x, y, [Which::First(2), Which::First(3), Which::Second(2), Which::Second(3)]);
        let a = simd::simd_swizzle!(temp0, z, [Which::First(0), Which::First(2), Which::Second(0), Which::Second(3)]);
        let b = simd::simd_swizzle!(temp0, z, [Which::First(1), Which::First(3), Which::Second(1), Which::Second(3)]);
        let c = simd::simd_swizzle!(temp1, z, [Which::First(0), Which::First(2), Which::Second(2), Which::Second(3)]);

        // handle the 4th row
        let d = a * f32x4::splat(self.0[3].0[0]);
        let d = d + (b * f32x4::splat(self.0[3].0[1]));
        let d = d + (c * f32x4::splat(self.0[3].0[2]));
        let d = f32x4::from_array([0.0, 0.0, 0.0, 1.0]) - d;

        Self([Vector4(a.to_array()), Vector4(b.to_array()), Vector4(c.to_array()), Vector4(d.to_array())])
    }

    /// A fast matrix inversion for general transformation matrices.
    #[rustfmt::skip]
    #[cfg(feature = "simd")]
    pub fn inversed_transform(&self) -> Self {
        let x = f32x4::from_array(self.0[0].0);
        let y = f32x4::from_array(self.0[1].0);
        let z = f32x4::from_array(self.0[2].0);

        // transpose the 3x3 sub-matrix
        let temp0 = simd::simd_swizzle!(x, y, [Which::First(0), Which::First(1), Which::Second(0), Which::Second(1)]);
        let temp1 = simd::simd_swizzle!(x, y, [Which::First(2), Which::First(3), Which::Second(2), Which::Second(3)]);
        let a = simd::simd_swizzle!(temp0, z, [Which::First(0), Which::First(2), Which::Second(0), Which::Second(3)]);
        let b = simd::simd_swizzle!(temp0, z, [Which::First(1), Which::First(3), Which::Second(1), Which::Second(3)]);
        let c = simd::simd_swizzle!(temp1, z, [Which::First(0), Which::First(2), Which::Second(2), Which::Second(3)]);

        // handle scale
        let sq = (a * a) + (b * b) + (c * c);
        // If sq[i] <= EPSILON, then sq[i] = 1.0, else sq[i] = (1.0 / sq[i])
        let is_small = sq.simd_le(f32x4::splat(f32::EPSILON));
        let sq = is_small.select(f32x4::splat(1.0), sq.recip());

        let a = a * sq;
        let b = b * sq;
        let c = c * sq;

        // handle the 4th row
        let d = a * f32x4::splat(self.0[3].0[0]);
        let d = d + (b * f32x4::splat(self.0[3].0[1]));
        let d = d + (c * f32x4::splat(self.0[3].0[2]));
        let d = f32x4::from_array([0.0, 0.0, 0.0, 1.0]) - d;

        Self([Vector4(a.to_array()), Vector4(b.to_array()), Vector4(c.to_array()), Vector4(d.to_array())])
    }

    /// General matrix inversion.
    #[rustfmt::skip]
    #[cfg(feature = "simd")]
    pub fn inversed(&self) -> Self {
        // 2x2 matrix multiplies
        #[inline]
        fn mat2_mul(a: f32x4, b: f32x4) -> f32x4 {
            let c = a * simd::simd_swizzle!(b, [0, 0, 3, 3]);
            let d = simd::simd_swizzle!(a, [2, 3, 0, 1]) * simd::simd_swizzle!(b, [1, 1, 2, 2]);
            c + d
        }

        #[inline]
        fn mat2_adj_mul(a: f32x4, b: f32x4) -> f32x4 {
            let c = simd::simd_swizzle!(a, [3, 0, 3, 0]) * b;
            let d = simd::simd_swizzle!(a, [2, 1, 2, 1]) * simd::simd_swizzle!(b, [1, 0, 3, 2]);
            c - d
        }

        #[inline]
        fn mat2_mul_adj(a: f32x4, b: f32x4) -> f32x4 {
            let c = a * simd::simd_swizzle!(b, [0, 0, 3, 3]);
            let d = simd::simd_swizzle!(a, [2, 3, 0, 1]) * simd::simd_swizzle!(b, [1, 1, 2, 2]);
            c - d
        }

        let x = f32x4::from_array(self.0[0].0);
        let y = f32x4::from_array(self.0[1].0);
        let z = f32x4::from_array(self.0[2].0);
        let w = f32x4::from_array(self.0[3].0);

        // sub-matrices
        let a = simd::simd_swizzle!(x, y, [Which::First(0), Which::First(1), Which::Second(0), Which::Second(1)]);
        let b = simd::simd_swizzle!(x, y, [Which::First(2), Which::First(3), Which::Second(2), Which::Second(3)]);
        let c = simd::simd_swizzle!(z, w, [Which::First(0), Which::First(1), Which::Second(0), Which::Second(1)]);
        let d = simd::simd_swizzle!(z, w, [Which::First(2), Which::First(3), Which::Second(2), Which::Second(3)]);

        // determinants
        let q = simd::simd_swizzle!(x, z, [Which::First(0), Which::First(2), Which::Second(0), Which::Second(2)]);
        let r = simd::simd_swizzle!(y, w, [Which::First(1), Which::First(3), Which::Second(1), Which::Second(3)]);
        let s = simd::simd_swizzle!(x, z, [Which::First(1), Which::First(3), Which::Second(1), Which::Second(3)]);
        let t = simd::simd_swizzle!(y, w, [Which::First(0), Which::First(2), Which::Second(0), Which::Second(2)]);
        let det_sub = (q * r) - (s * t);
        let det_a = simd::simd_swizzle!(det_sub, [0, 0, 0, 0]);
        let det_b = simd::simd_swizzle!(det_sub, [1, 1, 1, 1]);
        let det_c = simd::simd_swizzle!(det_sub, [2, 2, 2, 2]);
        let det_d = simd::simd_swizzle!(det_sub, [3, 3, 3, 3]);

        let dc = mat2_adj_mul(d, c);
        let ab = mat2_adj_mul(a, b);
        let xx = (det_d * a) - mat2_mul(b, dc);
        let ww = (det_a * d) - mat2_mul(c, ab);
        let det_m = det_a * det_d;

        let yy = (det_b * c) - mat2_mul_adj(d, ab);
        let zz = (det_c * b) - mat2_mul_adj(a, dc);
        let det_m = det_m + (det_b * det_c);

        // horizontal add packed
        #[inline]
        fn hadd(a: f32x4, b: f32x4) -> f32x4 {
            #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
            {
                use std::{mem, arch::x86_64};
                unsafe {
                    let a = mem::transmute(a);
                    let b = mem::transmute(b);
                    mem::transmute(x86_64::_mm_hadd_ps(a, b))
                }
            }
            #[cfg(any(not(target_arch = "x86_64"), not(target_feature = "sse3")))]
            {
                let a = simd::simd_swizzle!(a, [0, 2, 0, 2]);
                let b = simd::simd_swizzle!(b, [1, 3, 1, 3]);
                a + b
            }
        }
        let tr = ab * simd::simd_swizzle!(dc, [0, 2, 1, 3]);
        let tr = hadd(tr, tr);
        let tr = hadd(tr, tr);

        let det_m = det_m - tr;
        const ADJ_SIGN_MASK: f32x4 = f32x4::from_array([1.0, -1.0, -1.0, -1.0]);
        let det_m_recip = ADJ_SIGN_MASK / det_m;

        let xx = xx * det_m_recip;
        let yy = yy * det_m_recip;
        let zz = zz * det_m_recip;
        let ww = ww * det_m_recip;

        let a = simd::simd_swizzle!(xx, zz, [Which::First(3), Which::First(1), Which::Second(3), Which::Second(1)]);
        let b = simd::simd_swizzle!(xx, zz, [Which::First(2), Which::First(0), Which::Second(2), Which::Second(0)]);
        let c = simd::simd_swizzle!(yy, ww, [Which::First(3), Which::First(1), Which::Second(3), Which::Second(1)]);
        let d = simd::simd_swizzle!(yy, ww, [Which::First(2), Which::First(0), Which::Second(2), Which::Second(0)]);

        Self([Vector4(a.to_array()), Vector4(b.to_array()), Vector4(c.to_array()), Vector4(d.to_array())])
    }

    #[inline]
    #[rustfmt::skip]
    pub fn transposed(&self) -> Self {
        Self([
            Vector4([self.0[0].0[0], self.0[1].0[0], self.0[2].0[0], self.0[3].0[0]]),
            Vector4([self.0[0].0[1], self.0[1].0[1], self.0[2].0[1], self.0[3].0[1]]),
            Vector4([self.0[0].0[2], self.0[1].0[2], self.0[2].0[2], self.0[3].0[2]]),
            Vector4([self.0[0].0[3], self.0[1].0[3], self.0[2].0[3], self.0[3].0[3]]),
        ])
    }

    #[inline]
    pub fn narrowed(&self) -> Matrix3 {
        Matrix3([
            self.0[0].narrowed(),
            self.0[1].narrowed(),
            self.0[2].narrowed(),
        ])
    }
}

impl From<Quaternion> for Matrix4 {
    #[inline]
    #[rustfmt::skip]
    fn from(q: Quaternion) -> Self {
        // // https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/jay.htm
        // let q = f32x4::from_array(q.0.0);
        //
        // let a = simd::simd_swizzle!(q, [3, 2, 1, 0]);
        // let b = simd::simd_swizzle!(q, [2, 3, 0, 1]);
        // let c = simd::simd_swizzle!(q, [1, 0, 3, 2]);
        // let d = simd::simd_swizzle!(q, [0, 1, 2, 3]);
        //
        // // TODO(lavignes): could replace these masks with bitwise ops to flip the sign
        // let m1 = Self([
        //     Vector4((a * f32x4::from_array([1.0, 1.0, -1.0, 1.0])).to_array()),
        //     Vector4((b * f32x4::from_array([-1.0, 1.0, 1.0, 1.0])).to_array()),
        //     Vector4((c * f32x4::from_array([1.0, -1.0, 1.0, 1.0])).to_array()),
        //     Vector4((d * f32x4::from_array([-1.0, -1.0, -1.0, 1.0])).to_array()),
        // ]);
        //
        // let m2 = Self([
        //     Vector4((a * f32x4::from_array([1.0, 1.0, -1.0, -1.0])).to_array()),
        //     Vector4((b * f32x4::from_array([-1.0, 1.0, 1.0, -1.0])).to_array()),
        //     Vector4((c * f32x4::from_array([1.0, -1.0, 1.0, -1.0])).to_array()),
        //     Vector4((d * f32x4::from_array([1.0, 1.0, 1.0, 1.0])).to_array()),
        // ]);
        //
        // m1 * m2

        // LLVM Actually does a great job optimizing this. The unrolled matrix mul makes the other version seem less optimal
        Self([
            Vector4([2.0 * (q.0.0[0] * q.0.0[2] - q.0.0[3] * q.0.0[1]), 2.0 * (q.0.0[1] * q.0.0[2] + q.0.0[3] * q.0.0[0]), 1.0 - 2.0 * (q.0.0[0] * q.0.0[0] + q.0.0[1] * q.0.0[1]), 0.0]),
            Vector4([1.0 - 2.0 * (q.0.0[1] * q.0.0[1] + q.0.0[2] * q.0.0[2]), 2.0 * (q.0.0[0] * q.0.0[1] - q.0.0[3] * q.0.0[2]), 2.0 * (q.0.0[0] * q.0.0[2] + q.0.0[3] * q.0.0[1]), 0.0]),
            Vector4([2.0 * (q.0.0[0] * q.0.0[1] + q.0.0[3] * q.0.0[2]), 1.0 - 2.0 * (q.0.0[0] * q.0.0[0] + q.0.0[2] * q.0.0[2]), 2.0 * (q.0.0[1] * q.0.0[2] - q.0.0[3] * q.0.0[0]), 0.0]),
            Vector4([0.0, 0.0, 0.0, 1.0]),
        ])
    }
}

macro_rules! mat_mul {
    ($self_type:ty, $mat_type:ty) => {
        impl Mul<$mat_type> for $self_type {
            type Output = Matrix4;

            fn mul(self, rhs: $mat_type) -> Matrix4 {
                #[cfg(feature = "simd")]
                {
                    let a = f32x4::from_array(rhs.0[0].0);
                    let b = f32x4::from_array(rhs.0[1].0);
                    let c = f32x4::from_array(rhs.0[2].0);
                    let d = f32x4::from_array(rhs.0[3].0);
                    let mut ret = Matrix4::default();
                    for i in 0..4 {
                        let x = f32x4::splat(self.0[i].0[0]);
                        let y = f32x4::splat(self.0[i].0[1]);
                        let z = f32x4::splat(self.0[i].0[2]);
                        let w = f32x4::splat(self.0[i].0[3]);
                        ret.0[i] = Vector4(((x * a) + (y * b) + (z * c) + (w * d)).to_array());
                    }
                    ret
                }

                #[cfg(not(feature = "simd"))]
                {
                    let mut ret = Matrix4::default();
                    for i in 0..4 {
                        for j in 0..4 {
                            ret.0[i].0[0] += self.0[i].0[j] * rhs.0[j].0[0];
                            ret.0[i].0[1] += self.0[i].0[j] * rhs.0[j].0[1];
                            ret.0[i].0[2] += self.0[i].0[j] * rhs.0[j].0[2];
                            ret.0[i].0[3] += self.0[i].0[j] * rhs.0[j].0[3];
                        }
                    }
                    ret
                }
            }
        }
    };
}

mat_mul!(Matrix4, Matrix4);
mat_mul!(&Matrix4, Matrix4);
mat_mul!(Matrix4, &Matrix4);
mat_mul!(&Matrix4, &Matrix4);
