#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::borrow_deref_ref)]

extern crate tokenizers as tk;

mod decoders;
mod encoding;
mod error;
mod models;
mod normalizers;
mod pre_tokenizers;
mod processors;
mod token;
mod tokenizer;
mod trainers;
mod utils;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(target_family = "unix")]
static mut REGISTERED_FORK_CALLBACK: bool = false;

#[cfg(target_family = "unix")]
extern "C" fn child_after_fork() {
    use tk::parallelism::*;
    if has_parallelism_been_used() && !is_parallelism_configured() {
        println!(
            "huggingface/tokenizers: The current process just got forked, after parallelism has \
            already been used. Disabling parallelism to avoid deadlocks..."
        );
        println!("To disable this warning, you can either:");
        println!(
            "\t- Avoid using `tokenizers` before the fork if possible\n\
            \t- Explicitly set the environment variable {}=(true | false)",
            ENV_VARIABLE
        );
        set_parallelism(false);
    }
}

/// Tokenizers Module
#[pymodule]
pub fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    let _ = env_logger::try_init_from_env("TOKENIZERS_LOG");

    #[cfg(target_family = "unix")]
    unsafe {
        if !REGISTERED_FORK_CALLBACK {
            libc::pthread_atfork(None, None, Some(child_after_fork));
            REGISTERED_FORK_CALLBACK = true;
        }
    }

    m.add_class::<tokenizer::PyTokenizer>()?;
    m.add_class::<tokenizer::PyAddedToken>()?;
    m.add_class::<token::PyToken>()?;
    m.add_class::<encoding::PyEncoding>()?;
    m.add_class::<utils::PyRegex>()?;
    m.add_class::<utils::PyNormalizedString>()?;
    m.add_class::<utils::PyPreTokenizedString>()?;
    m.add_wrapped(wrap_pymodule!(models::models))?;
    m.add_wrapped(wrap_pymodule!(pre_tokenizers::pre_tokenizers))?;
    m.add_wrapped(wrap_pymodule!(decoders::decoders))?;
    m.add_wrapped(wrap_pymodule!(processors::processors))?;
    m.add_wrapped(wrap_pymodule!(normalizers::normalizers))?;
    m.add_wrapped(wrap_pymodule!(trainers::trainers))?;
    Ok(())
}
