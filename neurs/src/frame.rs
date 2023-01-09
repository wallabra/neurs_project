/*!
 * The [Frame] interfaces an Assembly with an external use case.
 *
 * This is how assmblies you compose using neurs interact with the
 * outside world. This can be anything, from small self-contained
 * applications and test cases, to video games.
 */

use crate::prelude::*;
use std::marker::PhantomData;

/// Parameters and specifics for how an Assembly is used and trained.
pub trait Frame<AssemblyType>
where
    AssemblyType: Assembly,
{
    type TrainHandle: FrameHandle<AssemblyType>;
    type ProdHandle: FrameHandle<AssemblyType>;

    /// Poll whether a slot for another run is available.
    fn can_run(&self) -> bool;

    /// Performs a training run.
    /// Returns a handle.
    fn start_train_run(&mut self, assembly: AssemblyType) -> Self::TrainHandle;

    /// Performs a production run.
    /// Returns a handle.
    fn start_run(&mut self, assembly: AssemblyType) -> Self::ProdHandle;
}

/// A simple Frame where a result is produced immediately and synchronously.
///
/// Use this for simple test cases that don't interface with the outside world
/// or with another complex system somehow.
pub trait SimpleFrame<AssemblyType>
where
    AssemblyType: Assembly,
{
    /// Run this frame for an <Assembly>.
    ///
    /// Returns a fitness value; if not applicable, just return zero.
    fn run(&mut self, assembly: AssemblyType) -> (AssemblyType, Result<f32, String>);

    fn _run_to_result(&mut self, assembly: AssemblyType) -> SimpleFrameHandle<AssemblyType> {
        let (asembly, result) = self.run(assembly);
        SimpleFrameHandle { assembly, result }
    }
}

pub struct SimpleFrameHandle<AssemblyType: Assembly> {
    assembly: AssemblyType,
    result: Result<f32, String>,
}

impl<AssemblyType: Assembly> FrameHandle<AssemblyType> for SimpleFrameHandle<AssemblyType> {
    fn ref_assembly(&self) -> &AssemblyType {
        &self.assembly
    }

    fn ref_assembly_mut(&mut self) -> &mut AssemblyType {
        &mut self.assembly
    }

    fn finish(self) -> AssemblyType {
        self.assembly
    }

    fn poll_state(&self) -> FrameRunState {
        use FrameRunState::*;

        if self.result.is_ok() {
            Done
        } else {
            Error(self.result.unwrap_err())
        }
    }

    fn get_fitness(&self) -> f32 {
        self.result.unwrap_or(0.0)
    }
}

impl<AssemblyType, AnySimpleFrame> Frame<AssemblyType> for AnySimpleFrame
where
    AnySimpleFrame: SimpleFrame<AssemblyType>,
    AssemblyType: Assembly,
{
    type TrainHandle = SimpleFrameHandle<AssemblyType>;
    type ProdHandle = SimpleFrameHandle<AssemblyType>;

    fn can_run(&self) -> bool {
        true
    }

    fn start_train_run(&mut self, assembly: AssemblyType) -> SimpleFrameHandle<AssemblyType> {
        self._run_to_result(assembly)
    }

    fn start_run(&mut self, assembly: AssemblyType) -> SimpleFrameHandle<AssemblyType> {
        self._run_to_result(assembly)
    }
}

#[derive(Default)]
pub enum FrameRunState {
    #[default]
    Waiting,

    Running,
    Done,
    Error(String),
}

impl FrameRunState {
    pub fn is_done(&self) -> bool {
        matches!(self, Done) || matches!(self, Error)
    }
}

/// A handle that tracks the state of a 'run' from a [Frame].
pub trait FrameHandle<AssemblyType>
where
    AssemblyType: Assembly,
{
    /// References the Assembly held by this handle.
    fn ref_assembly(&self) -> &AssemblyType;

    /// Mutably references the Assembly held by this handle.
    fn ref_assembly_mut(&mut self) -> &mut AssemblyType;

    /// Returns ownership of the Assembly held by this handle before dropping it.
    fn finish(self) -> AssemblyType;

    /// Polls the state of this handle.
    fn poll_state(&self) -> FrameRunState;

    /// Get the fitness value of this run.
    /// Return 0 if not applicable.
    fn get_fitness(&self) -> f32;
}

/// Holds a number of FrameHandles in it and helps keep track of them.
#[derive(Default)]
pub struct HandlePool<HandleType, AA>
where
    AA: Assembly,
    HandleType: FrameHandle<AA>,
{
    handles: Vec<HandleType>,
    _phantom: PhantomData<AA>,
}

#[derive(Default)]
pub struct HandleResult<AssemblyType>
where
    AssemblyType: Assembly,
{
    state: FrameRunState,
    fitness: f32,
    returned_assembly: Option<AssemblyType>,
}

impl<HandleType, AA> HandlePool<HandleType, AA>
where
    AA: Assembly,
    HandleType: FrameHandle<AA>,
{
    fn add_handle(&mut self, handle: HandleType) -> () {
        self.handles.push(handle);
    }

    fn poll_all(&mut self) -> Vec<HandleResult<AA>> {
        let res = vec![HandleResult::default(); self.handles.len()];

        let res = self
            .handles
            .iter()
            .zip(res.iter_mut())
            .map(|(&handle, &mut res)| {
                let state = handle.poll_state();
                res.state = state;

                if matches!(state, Done) {
                    res.fitness = handle.get_fitness();
                }
            });

        self.handles
            .drain_filter(|&h| h.poll_state().is_done())
            .zip(res.iter_mut())
            .map(|(handle, &mut res)| {
                if let handle = Some(handle) {
                    res.returned_assembly = handle.finish();
                }
            });

        res
    }
}

pub mod prelude {
    pub use super::{
        Frame, FrameHandle, FrameRunState, HandlePool, HandleResult, SimpleFrame, SimpleFrameHandle,
    };
}
