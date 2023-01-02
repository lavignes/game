use std::{
    any::{Any, TypeId},
    fmt::{self, Display, Formatter},
    iter, mem,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockResult},
};

use fnv::FnvHashMap;

use crate::ecs::{Fetch, Query, QueryError, QueryFetch, QueryIter, QueryParameters};

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Entity {
    index: usize,
    epoch: usize,
}

impl Display for Entity {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}:{})", self.index, self.epoch)
    }
}

pub trait Component: Send + Sync + 'static {}

impl<T> Component for T where T: Send + Sync + 'static {}

trait TypeErasedVec: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_empty(&self) -> Box<dyn TypeErasedVec>;
    fn swap_remove(&mut self, index: usize);
    fn transfer(&mut self, from: usize, vec: &mut dyn TypeErasedVec);
    fn clone_transfer(&mut self, from: usize) -> Box<dyn TypeErasedVec>;
}

#[inline]
fn reify_vec_lock<T: Component>(v: &Box<dyn TypeErasedVec>) -> &RwLock<Vec<T>> {
    v.as_any().downcast_ref::<RwLock<Vec<T>>>().unwrap()
}

#[inline]
fn reify_vec_mut<T: Component>(v: &mut dyn TypeErasedVec) -> &mut Vec<T> {
    v.as_any_mut()
        .downcast_mut::<RwLock<Vec<T>>>()
        .unwrap()
        .get_mut()
        .unwrap()
}

impl<T: Component> TypeErasedVec for RwLock<Vec<T>> {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    #[inline]
    fn clone_empty(&self) -> Box<dyn TypeErasedVec> {
        Box::new(RwLock::new(Vec::<T>::new()))
    }

    #[inline]
    fn swap_remove(&mut self, index: usize) {
        self.get_mut().unwrap().swap_remove(index);
    }

    #[inline]
    fn transfer(&mut self, from: usize, to: &mut dyn TypeErasedVec) {
        let value = self.get_mut().unwrap().swap_remove(from);
        reify_vec_mut(to).push(value);
    }

    #[inline]
    fn clone_transfer(&mut self, from: usize) -> Box<dyn TypeErasedVec> {
        let mut clone = self.clone_empty();
        self.transfer(from, clone.as_mut());
        clone
    }
}

struct Store {
    inner: Box<dyn TypeErasedVec>,
}

impl Store {
    #[inline]
    fn new<T: Component>() -> Store {
        Store {
            inner: Box::new(RwLock::new(Vec::<T>::new())),
        }
    }
}

#[inline]
const fn type_id_unwrap(type_id: TypeId) -> u64 {
    // TODO: If const `PartialOrd` is ever implemented for TypeId we can drop this.
    static_assertions::assert_eq_size!(TypeId, u64);
    unsafe { mem::transmute(type_id) }
}

struct TypeIdConstSorter<const N: usize>;

impl<const N: usize> TypeIdConstSorter<N> {
    #[inline]
    const fn sort(values: [(TypeId, usize); N]) -> [(TypeId, usize); N] {
        Self::quick_sort(values, 0, N as isize - 1)
    }

    #[inline]
    const fn type_ids(values: [(TypeId, usize); N]) -> [TypeId; N] {
        let mut output = [TypeId::of::<()>(); N];
        let mut i = 0;
        while i < N {
            output[i] = values[i].0;
            i += 1;
        }
        output
    }

    #[inline]
    const fn original_indices(values: [(TypeId, usize); N]) -> [usize; N] {
        let mut output = [0; N];
        let mut i = 0;
        while i < N {
            output[i] = values[i].1;
            i += 1;
        }
        output
    }

    #[inline]
    const fn quick_sort(
        mut values: [(TypeId, usize); N],
        mut low: isize,
        mut high: isize,
    ) -> [(TypeId, usize); N] {
        let range = high - low;
        if range <= 0 || range >= N as isize {
            return values;
        }
        loop {
            let mut i = low;
            let mut j = high;
            let p = values[(low + ((high - low) >> 1)) as usize].0;
            loop {
                while type_id_unwrap(values[i as usize].0) < type_id_unwrap(p) {
                    i += 1;
                }
                while type_id_unwrap(values[j as usize].0) > type_id_unwrap(p) {
                    j -= 1;
                }
                if i <= j {
                    if i != j {
                        let q = values[i as usize];
                        values[i as usize] = values[j as usize];
                        values[j as usize] = q;
                    }
                    i += 1;
                    j -= 1;
                }
                if i > j {
                    break;
                }
            }
            if j - low < high - i {
                if low < j {
                    values = Self::quick_sort(values, low, j);
                }
                low = i;
            } else {
                if i < high {
                    values = Self::quick_sort(values, i, high)
                }
                high = j;
            }
            if low >= high {
                break;
            }
        }
        values
    }
}

pub trait Tuple: Sized + Send + Sync + 'static {
    const TYPE_IDS: &'static [TypeId];
    const TYPE_IDS_SORTED: &'static [TypeId];
    const ORIGINAL_INDICES: &'static [usize];

    fn to_archetype(&self) -> Archetype;

    fn spawn(self, world: &mut World) -> Entity;
}

macro_rules! tuple_impl {
    ($(($name: ident, $index: tt)),*) => {
        impl<$($name: Component),*> Tuple for ($($name,)*) {
            const TYPE_IDS: &'static [TypeId] = &[$(TypeId::of::<$name>()),*];
            const TYPE_IDS_SORTED: &'static [TypeId] =
                &TypeIdConstSorter::type_ids(TypeIdConstSorter::sort([
                    $((TypeId::of::<$name>(), $index)),*
                ]));
            const ORIGINAL_INDICES: &'static [usize] =
                &TypeIdConstSorter::original_indices(TypeIdConstSorter::sort([
                    $((TypeId::of::<$name>(), $index)),*
                ]));

            fn to_archetype(&self) -> Archetype {
                // TODO: This sort is O(1) overhead, but could we sort the stores @ compile time?
                let mut original_ordered_stores = vec![
                    $(Some(Store::new::<$name>())),*
                ];

                // We want to make sure the stores are created in the sorted order
                let mut stores = Vec::with_capacity(Self::TYPE_IDS.len());
                for &i in Self::ORIGINAL_INDICES {
                    stores.push(original_ordered_stores[i].take().unwrap());
                }

                Archetype {
                    type_ids: Self::TYPE_IDS_SORTED.to_owned(),
                    stores,
                    entities: Vec::new(),
                }
            }

            fn spawn(self, world: &mut World) -> Entity {
                let archetype_index = if let Some(index) = world.types_to_archetypes.get(Self::TYPE_IDS_SORTED) {
                    *index
                } else {
                    let index = world.archetypes.len();
                    world.types_to_archetypes.insert(Self::TYPE_IDS_SORTED.to_owned(), index);
                    world.archetypes.push(self.to_archetype());
                    index
                };

                // We will place the entity at this index in the archetype
                let addr_index = world.archetypes[archetype_index].entities.len();

                let entity = if let Some(entity) = world.graveyard.pop() {
                    // Resurrect the entity. We increment the epoch to invalidate references.
                    let entity = Entity {
                        index: entity.index,
                        epoch: entity.epoch + 1,
                    };
                    world.addresses[entity.index] = EntityAddress {
                        archetype: archetype_index,
                        index: addr_index,
                        epoch: entity.epoch,
                    };
                    entity
                } else {
                    let entity = Entity {
                        index: world.addresses.len(),
                        epoch: 1,
                    };
                    world.addresses.push(EntityAddress {
                        archetype: archetype_index,
                        index: addr_index,
                        epoch: entity.epoch,
                    });
                    entity
                };

                world.archetypes[archetype_index].entities.push(entity);

                $(world.archetypes[archetype_index].push(Self::ORIGINAL_INDICES[$index], self.$index);)*

                entity
            }
        }
    };
}

tuple_impl!((A, 0));
tuple_impl!((A, 0), (B, 1));
tuple_impl!((A, 0), (B, 1), (C, 2));
tuple_impl!((A, 0), (B, 1), (C, 2), (D, 3));
tuple_impl!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4));
tuple_impl!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5));
tuple_impl!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5), (G, 6));
#[rustfmt::skip]
tuple_impl!((A, 0), (B, 1), (C, 2), (D, 3), (E, 4), (F, 5), (G, 6), (H, 7));

pub struct Archetype {
    type_ids: Vec<TypeId>,

    stores: Vec<Store>,
    pub(super) entities: Vec<Entity>,
}

impl Archetype {
    #[inline]
    fn push<T: Component>(&mut self, index: usize, t: T) {
        self.reify_store_mut(index).push(t)
    }

    #[inline]
    fn reify_store_mut<T: Component>(&mut self, index: usize) -> &mut Vec<T> {
        reify_vec_mut(&mut *self.stores[index].inner)
    }

    #[inline]
    pub(super) fn locate_type(&self, type_id: TypeId) -> Option<usize> {
        // Types are sorted, so we can binary search :-)
        self.type_ids
            .binary_search(&type_id)
            .map_or(None, |index| Some(index))
    }

    #[inline]
    pub(super) fn try_read_store<T: Component>(
        &self,
        index: usize,
    ) -> TryLockResult<RwLockReadGuard<Vec<T>>> {
        reify_vec_lock(&self.stores[index].inner).try_read()
    }

    #[inline]
    pub(super) fn try_write_store<T: Component>(
        &self,
        index: usize,
    ) -> TryLockResult<RwLockWriteGuard<Vec<T>>> {
        reify_vec_lock(&self.stores[index].inner).try_write()
    }

    fn swap_remove(&mut self, index: usize) -> Entity {
        for store in self.stores.iter_mut() {
            store.inner.swap_remove(index)
        }
        let entity = *self.entities.last().unwrap();
        self.entities.swap_remove(index);
        entity
    }
}

#[derive(Debug)]
struct EntityAddress {
    archetype: usize,
    index: usize,
    // last known epoch of the entity that lived here
    epoch: usize,
}

fn get_mut_2<'a, U, T: AsMut<[U]>>(
    slice: &'a mut T,
    index1: usize,
    index2: usize,
) -> (&'a mut U, &'a mut U) {
    let (left_index, right_index) = if index1 < index2 {
        (index1, index2)
    } else {
        (index2, index1)
    };
    let (left, right) = slice.as_mut().split_at_mut(left_index + 1);
    if index1 < index2 {
        (
            &mut left[left_index],
            &mut right[right_index - left_index - 1],
        )
    } else {
        (
            &mut right[right_index - left_index - 1],
            &mut left[left_index],
        )
    }
}

pub struct World {
    types_to_archetypes: FnvHashMap<Vec<TypeId>, usize>,
    pub(super) archetypes: Vec<Archetype>,
    addresses: Vec<EntityAddress>,
    graveyard: Vec<Entity>,
}

impl World {
    #[inline]
    pub fn new() -> World {
        World {
            types_to_archetypes: FnvHashMap::default(),
            archetypes: Vec::new(),
            addresses: Vec::new(),
            graveyard: Vec::new(),
        }
    }

    #[inline]
    pub fn spawn<T: Tuple>(&mut self, tuple: T) -> Entity {
        tuple.spawn(self)
    }

    pub fn despawn(&mut self, entity: Entity) -> Result<(), QueryError> {
        let addr = &mut self.addresses[entity.index];
        // Double-check that we aren't removing a stale entity reference
        if addr.epoch != entity.epoch {
            return Err(QueryError::EntityNotFound(entity));
        }

        addr.epoch += 1;
        let swapped = self.archetypes[addr.archetype].swap_remove(addr.index);
        self.graveyard.push(entity);

        // Because another entity was swapped into the index of the one we removed
        // we need to update the address to point to the new resident.
        self.addresses[swapped.index].index = addr.index;
        Ok(())
    }

    pub fn extend<T: Component>(&mut self, entity: Entity, value: T) -> Result<(), QueryError> {
        let addr = &mut self.addresses[entity.index];
        // Double-check that we aren't updating a stale entity reference
        if addr.epoch != entity.epoch {
            return Err(QueryError::EntityNotFound(entity));
        }

        let type_ids = {
            let old_archetype = &mut self.archetypes[addr.archetype];

            // The component already exists, replace it
            if let Some(index) = old_archetype.locate_type(TypeId::of::<T>()) {
                reify_vec_mut(&mut *old_archetype.stores[index].inner)[addr.index] = value;
                return Ok(());
            }

            // Get sorted type ids
            let mut type_ids: Vec<TypeId> = old_archetype
                .type_ids
                .iter()
                .cloned()
                .chain(iter::once(TypeId::of::<T>()))
                .collect();
            type_ids.sort_by(|&lhs, &rhs| type_id_unwrap(lhs).cmp(&type_id_unwrap(rhs)));
            type_ids
        };

        // Check if an archetype for these types already exists
        if let Some(&archetype_index) = self.types_to_archetypes.get(type_ids.as_slice()) {
            // We need to get a mutable ref to both archetypes, so we must break apart the archetypes
            // array into two parts.
            let (old_archetype, archetype) =
                get_mut_2(&mut self.archetypes, addr.archetype, archetype_index);

            // copy entity from old to new archetype
            for &type_id in &old_archetype.type_ids {
                let old_index = old_archetype.locate_type(type_id).unwrap();
                let index = archetype.locate_type(type_id).unwrap();
                old_archetype.stores[old_index]
                    .inner
                    .transfer(addr.index, &mut *archetype.stores[index].inner);
            }

            // insert new component
            let index = archetype.locate_type(TypeId::of::<T>()).unwrap();
            reify_vec_mut(&mut *archetype.stores[index].inner).push(value);

            // Update address
            addr.archetype = archetype_index;
            addr.index = archetype.entities.len() - 1;

            return Ok(());
        }

        // Finally, if all else fails: synthesize a new archetype
        let old_archetype = &mut self.archetypes[addr.archetype];

        // create a copy of all the old stores in their original sorted order
        let mut stores = Vec::new();
        for store in &mut old_archetype.stores {
            stores.push(Store {
                inner: store.inner.clone_transfer(addr.index),
            });
        }
        // then insert the new store at the correct position
        let index = type_ids
            .iter()
            .position(|&id| id == TypeId::of::<T>())
            .unwrap();
        let mut store = Store::new::<T>();
        reify_vec_mut(&mut *store.inner).push(value);
        stores.insert(index, store);

        // Update address
        addr.archetype = self.archetypes.len();
        addr.index = 0;

        self.types_to_archetypes
            .insert(type_ids.clone(), self.archetypes.len());
        self.archetypes.push(Archetype {
            entities: vec![entity],
            stores,
            type_ids,
        });

        Ok(())
    }

    pub fn remove<T: Component>(&mut self, entity: Entity) -> Result<(), QueryError> {
        let addr = &mut self.addresses[entity.index];
        // Double-check that we aren't updating a stale entity reference
        if addr.epoch != entity.epoch {
            return Err(QueryError::EntityNotFound(entity));
        }

        let type_ids = {
            let old_archetype = &mut self.archetypes[addr.archetype];
            if let Some(index) = old_archetype.locate_type(TypeId::of::<T>()) {
                let mut type_ids = old_archetype.type_ids.clone();
                type_ids.remove(index);
                type_ids
            } else {
                // Doesn't exist!
                return Ok(());
            }
        };

        // Check if an archetype for these types already exists
        if let Some(&archetype_index) = self.types_to_archetypes.get(type_ids.as_slice()) {
            // We need to get a mutable ref to both archetypes, so we must break apart the archetypes
            // array into two parts.
            let (old_archetype, archetype) =
                get_mut_2(&mut self.archetypes, addr.archetype, archetype_index);

            // copy entity from old to new archetype
            for &type_id in &type_ids {
                let old_index = old_archetype.locate_type(type_id).unwrap();
                let index = archetype.locate_type(type_id).unwrap();
                old_archetype.stores[old_index]
                    .inner
                    .transfer(addr.index, &mut *archetype.stores[index].inner);
            }

            // Remove uncopied component
            let index = old_archetype.locate_type(TypeId::of::<T>()).unwrap();
            old_archetype.stores[index].inner.swap_remove(addr.index);

            // Update address
            addr.archetype = archetype_index;
            addr.index = archetype.entities.len() - 1;

            return Ok(());
        }

        // Finally, if all else fails: synthesize a new archetype
        let old_archetype = &mut self.archetypes[addr.archetype];

        // create a copy of all the old stores sans the removed component
        let mut stores = Vec::new();
        for &type_id in &type_ids {
            let index = old_archetype.locate_type(type_id).unwrap();
            stores.push(Store {
                inner: old_archetype.stores[index].inner.clone_transfer(addr.index),
            });
        }

        // Remove uncopied component
        let index = old_archetype.locate_type(TypeId::of::<T>()).unwrap();
        old_archetype.stores[index].inner.swap_remove(addr.index);

        // Update address
        addr.archetype = self.archetypes.len();
        addr.index = 0;

        self.types_to_archetypes
            .insert(type_ids.clone(), self.archetypes.len());
        self.archetypes.push(Archetype {
            entities: vec![entity],
            stores,
            type_ids,
        });

        Ok(())
    }

    #[inline]
    pub fn query<T: QueryParameters>(&self) -> Result<Query<T>, QueryError> {
        Ok(QueryFetch::<T>::fetch(self)?.take().unwrap())
    }
}

#[cfg(test)]
mod test {
    use std::ops::Range;

    use super::*;
    use crate::ecs::Has;

    #[test]
    fn test() {
        let mut world = World::new();

        world.spawn(("guy 1".to_string(), 1));
        world.spawn(("guy 2".to_string(), 2));
        let a = world.spawn(("guy 3".to_string(), 3, [1, 2, 3, 4]));
        let b = world.spawn(("guy 4".to_string(), 4));

        for (a, b) in world.query::<(Entity, &String)>().unwrap().iter() {
            println!("{b} {a:?}");
        }

        println!("Despawning");
        world.despawn(a).unwrap();

        for (a, b, _) in world
            .query::<(Entity, &String, Has<String>)>()
            .unwrap()
            .iter()
        {
            println!("{b} {a:?}");
        }

        println!("Extending");
        world.extend(b, 1..2).unwrap();

        for (a, b, c, _) in world
            .query::<(Entity, &Range<i32>, &String, Has<i32>)>()
            .unwrap()
            .iter()
        {
            println!("{b:?} {c} {a:?}");
        }

        println!("Removing");
        world.remove::<Range<i32>>(b).unwrap();
        for (a, b, _) in world.query::<(Entity, &String, Has<i32>)>().unwrap().iter() {
            println!("{b} {a:?}");
        }
    }
}
