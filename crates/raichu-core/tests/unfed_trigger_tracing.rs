//! Compilation diagnostics captured in an isolated test process.
//!
//! A thread-local tracing subscriber shares callsite interest with every
//! thread in its process. Keep this binary's single capture test separate
//! from parallel model-compilation tests so an unobserved thread cannot
//! register the warning callsite with `Interest::never`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use raichu_core::CompiledModel;
#[path = "support/unfed_standby.rs"]
mod unfed_standby;
use unfed_standby::standby;

/// A `tracing` subscriber that keeps what was emitted, so the route the
/// diagnostic travels can be checked by running rather than by reading.
///
/// Written here rather than pulled in as a dependency: the workspace
/// installs no subscriber anywhere, which is the whole point of the
/// route (`tracing` costs nothing where nobody listens), and a test that
/// added one to the build would be measuring a different program.
mod collected {
    use std::fmt::Debug;
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Level, Metadata, Subscriber};

    /// The events one run emitted: level and formatted message.
    #[derive(Clone, Default)]
    pub struct Collected(Arc<Mutex<Vec<(Level, String)>>>);

    impl Collected {
        pub fn events(&self) -> Vec<(Level, String)> {
            self.0.lock().expect("collector poisoned").clone()
        }
    }

    #[derive(Default)]
    struct Message(String);

    impl Visit for Message {
        fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }

    impl Subscriber for Collected {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, event: &Event<'_>) {
            let mut message = Message::default();
            event.record(&mut message);
            self.0
                .lock()
                .expect("collector poisoned")
                .push((*event.metadata().level(), message.0));
        }
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }
}

#[test]
fn compilation_warns_for_an_unfed_trigger_and_stays_silent_when_wired() {
    let collected = collected::Collected::default();
    tracing::subscriber::with_default(collected.clone(), || {
        CompiledModel::compile(&standby("and", false, "up")).unwrap();
        let events = collected.events();
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].0, tracing::Level::WARN);
        assert!(events[0].1.starts_with("unfed trigger:"), "{events:?}");

        CompiledModel::compile(&standby("and", true, "down")).unwrap();
        assert_eq!(
            collected.events(),
            events,
            "a wired model emitted a diagnostic"
        );
    });
}
