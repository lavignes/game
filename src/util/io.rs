use std::{
    fs::File,
    io::{self, BufRead, BufReader, ErrorKind, Read, Seek},
    path::Path,
};

/// Easy way to return something that's error-like wrapped in an `std::io::Error`
#[inline]
pub fn io_err<T, E: Into<anyhow::Error>>(kind: ErrorKind, err: E) -> io::Result<T> {
    Err(io::Error::new(kind, err.into()))
}

/// Easy way to convert a result into an `std::io::Result`
#[inline]
pub fn io_err_result<T, E: Into<anyhow::Error>>(
    result: Result<T, E>,
    kind: ErrorKind,
) -> io::Result<T> {
    match result {
        Ok(t) => Ok(t),
        Err(err) => Err(io::Error::new(kind, err.into())),
    }
}

/// Map a `None` optional into an `std::io::Error`
#[inline]
pub fn io_err_option<T, E: Into<anyhow::Error>, F: Fn() -> E>(
    option: Option<T>,
    kind: ErrorKind,
    err: F,
) -> io::Result<T> {
    match option {
        Some(t) => Ok(t),
        None => Err(io::Error::new(kind, err().into())),
    }
}

#[inline]
pub fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    if reader.read(&mut bytes)? != bytes.len() {
        io_err(
            ErrorKind::UnexpectedEof,
            anyhow::anyhow!("Could not read enough bytes"),
        )
    } else {
        Ok(u32::from_le_bytes(bytes))
    }
}

#[inline]
pub fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut bytes = [0u8; 1];
    if reader.read(&mut bytes)? != bytes.len() {
        io_err(
            ErrorKind::UnexpectedEof,
            anyhow::anyhow!("Could not read enough bytes"),
        )
    } else {
        Ok(u8::from_le_bytes(bytes))
    }
}

/// Open a file at some path, returning a buffered reader.
///
/// Returns a more helpful error if the file cannot be opened.
pub fn buf_open<P: AsRef<Path>>(path: P) -> io::Result<impl BufRead + Seek> {
    let path: &Path = path.as_ref();
    let file = File::open(path).map_err(|err| {
        io::Error::new(
            ErrorKind::Other,
            anyhow::anyhow!("Could not open file {}: {}", path.display(), err),
        )
    })?;
    Ok(BufReader::new(file))
}
