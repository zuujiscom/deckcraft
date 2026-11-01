//! Crash recovery in the browser: every presentation with unsaved changes keeps a copy in
//! IndexedDB (craft-libs' `craft-web-recovery`: one store per tab session, held by a Web Lock so a
//! second tab leaves a running tab's copies alone, a write counted only once IndexedDB has
//! committed it). The desktop keeps its copies in a profile folder (`deckcraft_engine::recovery`),
//! which the browser doesn't have; the copies are the same native `.deckcraft` file.
//!
//! Startup reads every copy before the app is built and reopens them as unsaved presentations,
//! the newest last (so it is the one shown). A copy goes when its presentation is saved or closed.
//! Copies are written at most every [`EVERY`] seconds while a presentation changes, and at once
//! when the tab is hidden (the last moment a browser promises before it may discard the page).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use craft_web_recovery::{HiddenHook, Opened, Recovered, Snapshot, Store, UnloadGuard};
use deckcraft_engine::{DocState, Session};
use deckcraft_model::Presentation;
use deckcraft_ui_egui::SlideApp;

/// The least time between two rounds of copies (seconds): encoding runs on the page.
const EVERY: f64 = 10.0;
/// Presentations reopened per start (as on the desktop); the rest stay stored for the next one.
const MAX_REOPEN: usize = 32;

/// What [`open`] found, for the app to be built with.
pub struct Startup {
    pub store: Store,
    pub recovered: Vec<Recovered>,
    pub warnings: Vec<String>,
}

pub async fn open() -> Startup {
    let Opened { store, recovered, warnings } = craft_web_recovery::open("deckcraft").await;
    Startup { store, recovered, warnings: warnings.iter().map(ToString::to_string).collect() }
}

/// Reopen the copies in `session` as unsaved presentations → the store key each one keeps living
/// under (by document uid) and a message for the user.
pub fn restore(session: &mut Session, recovered: &[Recovered]) -> (HashMap<u64, String>, Option<String>) {
    let mut adopted = HashMap::new();
    let mut names = Vec::new();
    // `recovered` is newest first: open the newest last so it ends up active.
    for r in recovered.iter().take(MAX_REOPEN).rev() {
        let mut doc = match deckcraft_format::load(&r.bytes) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("recovery copy “{}” could not be opened: {e}", r.name);
                continue;
            }
        };
        deckcraft_format::repair(&mut doc);
        let title = if r.name.is_empty() { "Presentation" } else { r.name.as_str() };
        let mut st = DocState::new(doc, r.path.clone(), format!("{title} (Recovered)"));
        // Unsaved: nothing the user has saved matches it.
        st.saved_doc = Arc::new((*st.doc).clone());
        st.revision += 1;
        let uid = st.uid;
        session.add_document(st);
        adopted.insert(uid, r.key.clone());
        names.push(title.to_string());
    }
    let message = match names.as_slice() {
        [] => None,
        [one] => Some(format!("Recovered “{one}” from this browser. Save it to keep it.")),
        many => Some(format!("Recovered {} presentations from this browser. Save them to keep them.", many.len())),
    };
    (adopted, message)
}

/// One presentation's copy.
struct Entry {
    key: String,
    /// The version last handed to the store (compared by pointer: every edit makes a new one).
    queued: Option<Arc<Presentation>>,
}

/// An unsaved presentation as of the last frame, enough to write its copy without the app (the
/// hidden-tab hook runs outside eframe).
struct Dirty {
    uid: u64,
    revision: u64,
    doc: Arc<Presentation>,
    title: String,
    path: Option<String>,
}

struct State {
    store: Store,
    entries: HashMap<u64, Entry>,
    dirty: Vec<Dirty>,
}

impl State {
    /// Hand the store a copy of every unsaved presentation it doesn't hold yet.
    fn write(&mut self) {
        let State { store, entries, dirty } = self;
        for d in dirty.iter() {
            if entries.get(&d.uid).and_then(|e| e.queued.as_ref()).is_some_and(|q| Arc::ptr_eq(q, &d.doc)) {
                continue;
            }
            let bytes = match deckcraft_format::save(&d.doc) {
                Ok(b) => b,
                Err(e) => {
                    log::warn!("recovery copy not encoded: {e}");
                    continue;
                }
            };
            let entry = entries.entry(d.uid).or_insert_with(|| Entry { key: store.key(d.uid), queued: None });
            match store.put(&entry.key, d.revision, Snapshot { name: &d.title, path: d.path.as_deref(), meta: None, bytes: &bytes }) {
                Ok(()) => entry.queued = Some(d.doc.clone()),
                Err(e) => log::warn!("recovery copy not stored: {e}"),
            }
        }
    }

    fn pending(&self) -> bool {
        self.dirty.iter().any(|d| self.entries.get(&d.uid).and_then(|e| e.queued.as_ref()).is_none_or(|q| !Arc::ptr_eq(q, &d.doc)))
    }
}

/// The copies of the open presentations, kept up to date once per frame, plus the "Leave site?"
/// question while anything is unsaved.
pub struct WebRecovery {
    state: Rc<RefCell<State>>,
    unsaved: Rc<Cell<bool>>,
    last: f64,
    _hidden: Option<HiddenHook>,
    _unload: Option<UnloadGuard>,
}

impl WebRecovery {
    pub fn new(store: Store, adopted: HashMap<u64, String>, ctx: &egui::Context) -> Self {
        let wake = ctx.clone();
        store.set_waker(move || wake.request_repaint());
        let entries = adopted.into_iter().map(|(uid, key)| (uid, Entry { key, queued: None })).collect();
        let state = Rc::new(RefCell::new(State { store, entries, dirty: Vec::new() }));
        let unsaved = Rc::new(Cell::new(false));
        let hidden_state = state.clone();
        let hidden = HiddenHook::new(move || {
            if let Ok(mut s) = hidden_state.try_borrow_mut() {
                s.write();
            }
        })
        .map_err(|e| log::warn!("recovery: no hidden-tab hook: {e}"))
        .ok();
        let guard_unsaved = unsaved.clone();
        let unload = UnloadGuard::new(move || guard_unsaved.get()).map_err(|e| log::warn!("recovery: no unload guard: {e}")).ok();
        Self { state, unsaved, last: f64::NEG_INFINITY, _hidden: hidden, _unload: unload }
    }

    pub fn tick(&mut self, app: &SlideApp, ctx: &egui::Context) {
        let mut s = self.state.borrow_mut();
        for w in s.store.take_written() {
            if let Err(err) = w.result
                && let Some(e) = s.entries.values_mut().find(|e| e.key == w.key)
            {
                log::warn!("recovery copy not written: {err}");
                // Try again at the next round.
                e.queued = None;
            }
        }
        let docs = app.session.documents();
        s.dirty = docs
            .iter()
            .filter(|d| d.is_dirty())
            .map(|d| Dirty { uid: d.uid, revision: d.revision, doc: d.doc.clone(), title: d.title(), path: d.path.clone() })
            .collect();
        self.unsaved.set(!s.dirty.is_empty());
        // Saved presentations and closed ones have no copy.
        let gone: Vec<u64> = s.entries.keys().copied().filter(|uid| !s.dirty.iter().any(|d| d.uid == *uid)).collect();
        for uid in gone {
            if let Some(e) = s.entries.remove(&uid) {
                s.store.remove(&e.key);
            }
        }
        if !s.pending() {
            return;
        }
        let now = ctx.input(|i| i.time);
        if now - self.last < EVERY {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(EVERY - (now - self.last)));
            return;
        }
        self.last = now;
        s.write();
    }
}
