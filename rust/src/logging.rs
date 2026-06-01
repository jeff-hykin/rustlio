//! Console logging built on the standard [`log`] facade + [`env_logger`].
//!
//! Verbosity is driven by [`LogLevel`] (from the YAML `log_level` field):
//!
//! | config     | `log` filter | shows                                    |
//! |------------|--------------|------------------------------------------|
//! | `silent`   | Off          | nothing                                  |
//! | `normal`   | Info         | key progress, results, warnings, errors  |
//! | `detailed` | Debug        | + per-frame progress and extra detail    |
//! | `verbose`  | Trace        | + low-level per-message dumps            |
//!
//! Set the `RUST_LOG` environment variable to override the config (standard
//! `env_logger` syntax, e.g. `RUST_LOG=debug`).
//!
//! Output goes to stdout with a minimal format (plain message at info, tagged
//! for other levels) so downstream tooling can capture results simply.

use crate::commons::LogLevel;
use std::io::Write;

/// Initialize the global logger from `level`. Idempotent and safe to call once
/// per process; a `RUST_LOG` env var (if set) takes precedence over `level`.
pub fn init(level: LogLevel) {
    let mut builder = env_logger::Builder::new();
    builder
        .filter_level(level.to_level_filter())
        .target(env_logger::Target::Stdout)
        .format(|buf, record| match record.level() {
            log::Level::Info => writeln!(buf, "{}", record.args()),
            log::Level::Warn => writeln!(buf, "warning: {}", record.args()),
            log::Level::Error => writeln!(buf, "error: {}", record.args()),
            log::Level::Debug => writeln!(buf, "[debug] {}", record.args()),
            log::Level::Trace => writeln!(buf, "[trace] {}", record.args()),
        });
    if std::env::var("RUST_LOG").is_ok() {
        builder.parse_env("RUST_LOG");
    }
    let _ = builder.try_init();
}
