use std::{
    any::{self, TypeId},
    iter,
    iter::{Repeat, Take, Zip},
    marker::PhantomData,
    slice,
    sync::{RwLockReadGuard, RwLockWriteGuard},
};

use crate::ecs::{Archetype, Component, Entity, World};

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("Entity: {0} is not found")]
    EntityNotFound(Entity),

    #[error("The component: {0} is locked by another query")]
    Locked(&'static str),
}

pub trait QueryStoreLock<'world> {
    type Item;

    fn lock(world: &'world World, archetype: usize, index: usize)
        -> Result<Self::Item, QueryError>;
}

pub struct QueryStoreReadLock<T> {
    phantom: PhantomData<T>,
}

impl<'world, T: Component> QueryStoreLock<'world> for QueryStoreReadLock<T> {
    type Item = RwLockReadGuard<'world, Vec<T>>;

    #[inline]
    fn lock(
        world: &'world World,
        archetype: usize,
        index: usize,
    ) -> Result<Self::Item, QueryError> {
        let archetype = &world.archetypes[archetype];

        if let Ok(read_guard) = archetype.try_read_store(index) {
            Ok(read_guard)
        } else {
            Err(QueryError::Locked(any::type_name::<T>()))
        }
    }
}

pub struct QueryStoreWriteLock<T> {
    phantom: PhantomData<T>,
}

impl<'world, T: Component> QueryStoreLock<'world> for QueryStoreWriteLock<T> {
    type Item = RwLockWriteGuard<'world, Vec<T>>;

    #[inline]
    fn lock(
        world: &'world World,
        archetype: usize,
        index: usize,
    ) -> Result<Self::Item, QueryError> {
        let archetype = &world.archetypes[archetype];

        if let Ok(write_guard) = archetype.try_write_store(index) {
            Ok(write_guard)
        } else {
            Err(QueryError::Locked(any::type_name::<T>()))
        }
    }
}

pub trait QueryParameter {
    type QueryStoreLock: for<'world> QueryStoreLock<'world>;

    fn location(archetype: &Archetype) -> Option<usize>;
}

impl<T: Component> QueryParameter for &T {
    type QueryStoreLock = QueryStoreReadLock<T>;

    #[inline]
    fn location(archetype: &Archetype) -> Option<usize> {
        archetype.locate_type(TypeId::of::<T>())
    }
}

impl<T: Component> QueryParameter for &mut T {
    type QueryStoreLock = QueryStoreWriteLock<T>;

    #[inline]
    fn location(archetype: &Archetype) -> Option<usize> {
        archetype.locate_type(TypeId::of::<T>())
    }
}

impl<T: Component> QueryParameter for Option<&T> {
    type QueryStoreLock = QueryStoreReadLock<T>;

    #[inline]
    fn location(archetype: &Archetype) -> Option<usize> {
        archetype.locate_type(TypeId::of::<T>())
    }
}

impl<T: Component> QueryParameter for Option<&mut T> {
    type QueryStoreLock = QueryStoreWriteLock<T>;

    #[inline]
    fn location(archetype: &Archetype) -> Option<usize> {
        archetype.locate_type(TypeId::of::<T>())
    }
}

impl<'world> QueryStoreLock<'world> for Entity {
    type Item = &'world Vec<Entity>;

    #[inline]
    fn lock(
        world: &'world World,
        archetype: usize,
        _index: usize,
    ) -> Result<Self::Item, QueryError> {
        Ok(&world.archetypes[archetype].entities)
    }
}

impl QueryParameter for Entity {
    type QueryStoreLock = Self;

    #[inline]
    fn location(_archetype: &Archetype) -> Option<usize> {
        Some(0)
    }
}

pub struct Has<T> {
    phantom: PhantomData<T>,
}

pub struct HasIter {
    inner: Take<Repeat<()>>,
    count: usize,
}

impl HasIter {
    #[inline]
    pub fn new(count: usize) -> Self {
        Self {
            inner: iter::repeat(()).take(count),
            count,
        }
    }
}

impl Clone for HasIter {
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.count)
    }
}

impl Iterator for HasIter {
    type Item = ();

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'world, T: Component> QueryStoreLock<'world> for Has<T> {
    type Item = HasIter;

    #[inline]
    fn lock(
        world: &'world World,
        archetype: usize,
        _index: usize,
    ) -> Result<Self::Item, QueryError> {
        Ok(HasIter::new(world.archetypes[archetype].entities.len()))
    }
}

impl<T: Component> QueryParameter for Has<T> {
    type QueryStoreLock = Self;

    #[inline]
    fn location(archetype: &Archetype) -> Option<usize> {
        archetype.locate_type(TypeId::of::<T>())
    }
}

pub trait QueryParameters: for<'world> QueryStoreLock<'world> {}

macro_rules! query_parameters_impl {
    ($($name: ident),*) => {
        #[allow(unused_parens)]
        impl<'world, $($name: QueryParameter,)*> QueryParameters for ($($name,)*) {}

        #[allow(unused_parens)]
        #[allow(non_snake_case)]
        impl<'world, $($name: QueryParameter,)*> QueryStoreLock<'world> for ($($name,)*) {
            #[allow(unused_parens)]
            type Item = Vec<($(<$name::QueryStoreLock as QueryStoreLock<'world>>::Item),*)>;

            fn lock(world: &'world World, _archetype: usize, _index: usize) -> Result<Self::Item, QueryError> {
                let mut result = Vec::new();
                for (i, archetype) in world.archetypes.iter().enumerate() {
                    // Get the indices of the stores up-front and save them
                    $(let $name = $name::location(&archetype);)*

                    if $($name.is_some())&&* {
                        result.push(($(<$name::QueryStoreLock as QueryStoreLock<'world>>::lock(world, i, $name.unwrap())?),*));
                    }
                }
                Ok(result)
            }
        }
    };
}

query_parameters_impl!(A);
query_parameters_impl!(A, B);
query_parameters_impl!(A, B, C);
query_parameters_impl!(A, B, C, D);
query_parameters_impl!(A, B, C, D, E);
query_parameters_impl!(A, B, C, D, E, F);
query_parameters_impl!(A, B, C, D, E, F, G);
query_parameters_impl!(A, B, C, D, E, F, G, H);

pub trait FetchItem<'item> {
    type Inner;

    fn inner(&'item mut self) -> Self::Inner;
}

pub(super) trait Fetch<'world> {
    type Item: for<'item> FetchItem<'item>;

    fn fetch(world: &'world World) -> Result<Self::Item, QueryError>;
}

pub(super) struct QueryFetch<T> {
    phantom: PhantomData<T>,
}

impl<'world, T: QueryParameters> Fetch<'world> for QueryFetch<T> {
    type Item = Option<Query<'world, T>>;

    #[inline]
    fn fetch(world: &'world World) -> Result<Self::Item, QueryError> {
        Ok(Some(Query {
            _world: world,
            view: T::lock(world, 0, 0)?,
        }))
    }
}

pub struct Query<'world, T: QueryParameters> {
    _world: &'world World,
    view: <T as QueryStoreLock<'world>>::Item,
}

impl<'item, 'world, T: QueryParameters> FetchItem<'item> for Option<Query<'world, T>> {
    type Inner = Query<'world, T>;

    #[inline]
    fn inner(&'item mut self) -> Self::Inner {
        self.take().unwrap()
    }
}

impl<'item, 'world, T: 'item> FetchItem<'item> for RwLockReadGuard<'world, T> {
    type Inner = &'item T;

    #[inline]
    fn inner(&'item mut self) -> Self::Inner {
        self
    }
}

impl<'item, 'world, T: 'item> FetchItem<'item> for RwLockWriteGuard<'world, T> {
    type Inner = &'item mut T;

    #[inline]
    fn inner(&'item mut self) -> Self::Inner {
        &mut *self
    }
}

impl<'item> FetchItem<'item> for &Vec<Entity> {
    type Inner = Self;

    #[inline]
    fn inner(&'item mut self) -> Self::Inner {
        self
    }
}

impl<'item> FetchItem<'item> for HasIter {
    type Inner = &'item Self;

    #[inline]
    fn inner(&'item mut self) -> Self::Inner {
        self
    }
}

macro_rules! impl_zip {
    ($name: ident, $zip_type: ty, $mapping: expr, $($T: ident),*) => {
        pub struct $name<A: Iterator, $($T: Iterator,)*> {
            inner: $zip_type,
        }

        impl<A: Iterator, $($T: Iterator,)*> $name<A, $($T,)*> {
            #[inline]
            #[allow(non_snake_case)]
            pub fn new(A: A, $($T: $T,)*) -> Self {
                Self {
                    inner: A$(.zip($T))*
                }
            }
        }

        impl<A: Iterator, $($T: Iterator,)*> Iterator for $name<A, $($T,)*> {
            type Item = (A::Item, $($T::Item,)*);

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.inner.next().map($mapping)
            }
        }
    };
}

#[rustfmt::skip]
impl_zip!(Zip3, Zip<Zip<A, B>, C>, |((a, b), c)| {(a, b, c)}, B, C);
#[rustfmt::skip]
impl_zip!(Zip4, Zip<Zip<Zip<A, B>, C>, D>, |(((a, b), c), d)| {(a, b, c, d)}, B, C, D);
#[rustfmt::skip]
impl_zip!(Zip5, Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, |((((a, b), c), d), e)| {(a, b, c, d, e)}, B, C, D, E);
#[rustfmt::skip]
impl_zip!(Zip6, Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, |(((((a, b), c), d), e), f)| {(a, b, c, d, e, f)}, B, C, D, E, F);
#[rustfmt::skip]
impl_zip!(Zip7, Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, |((((((a, b), c), d), e), f), g)| {(a, b, c, d, e, f, g)}, B, C, D, E, F, G);
#[rustfmt::skip]
impl_zip!(Zip8, Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, |(((((((a, b), c), d), e), f), g), h)| {(a, b, c, d, e, f, g, h)}, B, C, D, E, F, G, H);

pub struct ChainedIterator<I> {
    current_iter: Option<I>,
    iterators: Vec<I>,
}

impl<I: Iterator> ChainedIterator<I> {
    #[doc(hidden)]
    pub fn new(mut iterators: Vec<I>) -> Self {
        let current_iter = iterators.pop();
        Self {
            current_iter,
            iterators,
        }
    }
}

impl<I: Iterator> Iterator for ChainedIterator<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.current_iter {
            Some(ref mut iter) => match iter.next() {
                None => {
                    self.current_iter = self.iterators.pop();
                    if let Some(ref mut iter) = self.current_iter {
                        iter.next()
                    } else {
                        None
                    }
                }
                item => item,
            },
            None => None,
        }
    }
}

type QueryParameterItem<'world, Q> =
    <<Q as QueryParameter>::QueryStoreLock as QueryStoreLock<'world>>::Item;

pub trait QueryIter<'iter> {
    type Iter: Iterator;

    fn iter(&'iter mut self) -> Self::Iter;
}

type QueryParameterIter<'iter, 'world, A> =
    <QueryParameterItem<'world, A> as QueryIter<'iter>>::Iter;

impl<'iter, 'world, T: Component> QueryIter<'iter> for RwLockReadGuard<'world, Vec<T>> {
    type Iter = slice::Iter<'iter, T>;

    #[inline]
    fn iter(&'iter mut self) -> Self::Iter {
        <[T]>::iter(self)
    }
}

impl<'iter, 'world, T: Component> QueryIter<'iter> for RwLockWriteGuard<'world, Vec<T>> {
    type Iter = slice::IterMut<'iter, T>;

    #[inline]
    fn iter(&'iter mut self) -> Self::Iter {
        <[T]>::iter_mut(self)
    }
}

impl<'iter> QueryIter<'iter> for &Vec<Entity> {
    type Iter = slice::Iter<'iter, Entity>;

    #[inline]
    fn iter(&'iter mut self) -> Self::Iter {
        <[Entity]>::iter(self)
    }
}

impl<'iter> QueryIter<'iter> for HasIter {
    type Iter = Self;

    #[inline]
    fn iter(&'iter mut self) -> Self::Iter {
        self.clone()
    }
}

impl<'iter, 'world, A: QueryParameter> QueryIter<'iter> for Query<'world, (A,)>
where
    QueryParameterItem<'world, A>: QueryIter<'iter>,
{
    type Iter = ChainedIterator<QueryParameterIter<'iter, 'world, A>>;

    #[inline]
    fn iter(&'iter mut self) -> Self::Iter {
        ChainedIterator::new(self.view.iter_mut().map(|i| i.iter()).collect())
    }
}

impl<'iter, 'world, A: QueryParameter, B: QueryParameter> QueryIter<'iter> for Query<'world, (A, B)>
where
    QueryParameterItem<'world, A>: QueryIter<'iter>,
    QueryParameterItem<'world, B>: QueryIter<'iter>,
{
    type Iter = ChainedIterator<
        Zip<QueryParameterIter<'iter, 'world, A>, QueryParameterIter<'iter, 'world, B>>,
    >;

    #[inline]
    fn iter(&'iter mut self) -> Self::Iter {
        ChainedIterator::new(
            self.view
                .iter_mut()
                .map(|(a, b)| a.iter().zip(b.iter()))
                .collect(),
        )
    }
}

macro_rules! query_iter {
    ($zip_type: ident, $($name: ident),*) => {
        #[allow(non_snake_case)]
        impl<'iter, 'world, $($name: QueryParameter),*> QueryIter<'iter> for Query<'world, ($($name,)*)>
        where
            $(QueryParameterItem<'world, $name>: QueryIter<'iter>),*
        {
            type Iter = ChainedIterator<$zip_type<$(QueryParameterIter<'iter, 'world, $name>,)*>>;

            #[inline]
            fn iter(&'iter mut self) -> Self::Iter {
                ChainedIterator::new(
                    self.view
                        .iter_mut()
                        .map(|($(ref mut $name,)*)| $zip_type::new($($name.iter(),)*))
                        .collect()
                )
            }
        }
    }
}

query_iter!(Zip3, A, B, C);
query_iter!(Zip4, A, B, C, D);
query_iter!(Zip5, A, B, C, D, E);
query_iter!(Zip6, A, B, C, D, E, F);
query_iter!(Zip7, A, B, C, D, E, F, G);
query_iter!(Zip8, A, B, C, D, E, F, G, H);
