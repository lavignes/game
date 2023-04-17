use std::{
    io::{self, ErrorKind, Read, Seek, SeekFrom},
    slice::Iter,
    str,
};

use crate::{math::Vector2, util};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum TextureFormat {
    Dxt1,
}

#[derive(Debug, Default)]
struct RawMipLevel {
    start: usize,
    end: usize,
    size: Vector2,
    bytes_per_row: usize,
}

#[derive(Debug, Default)]
pub struct MipLevel<'a> {
    data: &'a [u8],
    size: Vector2,
    bytes_per_row: usize,
}

impl<'a> MipLevel<'a> {
    #[inline]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    #[inline]
    pub fn size(&self) -> Vector2 {
        self.size
    }

    #[inline]
    pub fn bytes_per_row(&self) -> usize {
        self.bytes_per_row
    }
}

pub struct Texture {
    data: Vec<u8>,
    format: TextureFormat,
    mip_levels: Vec<RawMipLevel>,
}

impl Texture {
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            format: TextureFormat::Dxt1,
            mip_levels: Vec::new(),
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.mip_levels.clear();
    }

    #[inline]
    pub fn mip_levels(&self) -> MipLevelIterator {
        MipLevelIterator {
            inner: self.mip_levels.iter(),
            data: &self.data,
        }
    }

    #[inline]
    pub fn format(&self) -> TextureFormat {
        self.format
    }
}

pub struct MipLevelIterator<'a> {
    inner: Iter<'a, RawMipLevel>,
    data: &'a Vec<u8>,
}

impl<'a> Iterator for MipLevelIterator<'a> {
    type Item = MipLevel<'a>;

    #[inline]
    fn next(&mut self) -> Option<MipLevel<'a>> {
        self.inner.next().map(|level| MipLevel {
            data: &self.data[level.start..level.end],
            size: level.size,
            bytes_per_row: level.bytes_per_row,
        })
    }
}

pub struct DDSReader {}

bitflags::bitflags! {
    struct PixelFormatFlags: u32 {
        const ALPHA_PIXELS = 0x00000001;
        const FOUR_CHARACTER_CODE = 0x00000004;
        const RGB = 0x00000040;
        const LUMINANCE = 0x00020000;
    }
}

bitflags::bitflags! {
    struct CapabilityFlags: u32 {
        const COMPLEX = 0x00000008;
        const TEXTURE = 0x00001000;
        const MIPMAP = 0x00400000;
    }
}

impl DDSReader {
    #[inline]
    pub fn new() -> Self {
        Self {}
    }

    pub fn read_into<R: Read + Seek>(
        &mut self,
        reader: &mut R,
        texture: &mut Texture,
    ) -> io::Result<()> {
        reader.seek(SeekFrom::Start(0x00))?;

        let expected_magic = u32::from_le_bytes([b'D', b'D', b'S', b' ']);

        let magic = util::read_u32(reader)?;
        if magic != expected_magic {
            return util::io_err(
                ErrorKind::InvalidData,
                anyhow::anyhow!(
                    "Expected a 'DDS ' ({expected_magic:04X}) instead found {magic:04X}",
                ),
            );
        }

        reader.seek(SeekFrom::Start(0x0C))?;
        let height = util::read_u32(reader)?;
        let width = util::read_u32(reader)?;
        let _pitch = util::read_u32(reader)?;
        reader.seek(SeekFrom::Current(0x04))?;
        let mip_levels = util::read_u32(reader)?;

        reader.seek(SeekFrom::Start(0x50))?;
        let format_flags_bytes = util::read_u32(reader)?;
        let format_flags = util::io_err_option(
            PixelFormatFlags::from_bits(format_flags_bytes),
            ErrorKind::InvalidData,
            || {
                anyhow::anyhow!(
                    "Unsupported DDS pixel format ({format_flags_bytes:04X}). The file is probably malformed",
                )
            },
        )?;
        let four_character_code_bytes = util::read_u32(reader)?.to_le_bytes();
        let four_character_code = util::io_err_result(
            str::from_utf8(&four_character_code_bytes),
            ErrorKind::InvalidData,
        )?;
        // TODO: Don't I want to use these bitmasks for something?
        //   It looks like I could support non 32-bit RGB formats
        let _rgb_bit_counts = util::read_u32(reader)?;
        let _r_bit_mask = util::read_u32(reader)?.to_le_bytes();
        let _g_bit_mask = util::read_u32(reader)?.to_le_bytes();
        let _b_bit_mask = util::read_u32(reader)?.to_le_bytes();
        let _a_bit_mask = util::read_u32(reader)?.to_le_bytes();
        let capabilities_bytes = util::read_u32(reader)?;
        util::io_err_option(
            CapabilityFlags::from_bits(capabilities_bytes),
            ErrorKind::InvalidData,
            || {
                anyhow::anyhow!(
                    "Unsupported DDS capabilities ({capabilities_bytes:04X}). The file is probably malformed",
                )
            },
        )?;

        // Jump to pixel data (it is further down on FourCharacterCode == "DX11" but we dont do it)
        reader.seek(SeekFrom::Start(0x70))?;
        if format_flags.contains(PixelFormatFlags::FOUR_CHARACTER_CODE) {
            let block_size;
            match four_character_code {
                "DXT1" => {
                    texture.format = TextureFormat::Dxt1;
                    block_size = 8;
                }
                _ => {
                    return util::io_err(
                        ErrorKind::InvalidData,
                        anyhow::anyhow!("Unsupported compression format: {}", four_character_code),
                    );
                }
            }
            let mut offset = 0;
            for mip_level in 0..mip_levels {
                let mip_width = (width >> mip_level) as usize;
                let mip_height = (height >> mip_level) as usize;
                let mip_pitch = ((mip_width + 3) / 4).max(1) * block_size;
                // This *should* also be the same as pitch at mip_level==0
                let linear_size = mip_pitch * ((mip_height + 3) / 4).max(1);
                texture.data.reserve(linear_size);
                for _ in 0..linear_size {
                    texture.data.push(util::read_u8(reader)?);
                }
                texture.mip_levels.push(RawMipLevel {
                    start: offset,
                    end: offset + linear_size,
                    size: (mip_width as f32, mip_height as f32).into(),
                    bytes_per_row: mip_pitch,
                });
                offset += linear_size;
            }
        } else {
            return util::io_err(
                ErrorKind::InvalidData,
                anyhow::anyhow!("Unsupported DDS pixel format {format_flags_bytes:04X}"),
            );
        }

        Ok(())
    }
}
