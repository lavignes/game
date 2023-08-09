mod io;

use std::{
    collections::{HashMap, HashSet},
    hash::{BuildHasherDefault, Hasher},
};

pub use io::*;

pub type FastHashMap<K, V> = HashMap<K, V, BuildHasherDefault<DjbHasher>>;

pub type FastHashSet<T> = HashSet<T, BuildHasherDefault<DjbHasher>>;

pub struct DjbHasher(u64);

impl Default for DjbHasher {
    #[inline]
    fn default() -> Self {
        Self(5381)
    }
}

impl Hasher for DjbHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &c in bytes {
            self.0 = (self.0 << 5).wrapping_add(self.0).wrapping_add(c as u64)
        }
    }
}
