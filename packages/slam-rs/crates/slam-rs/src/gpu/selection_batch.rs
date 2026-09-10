//! A single scanner's keys carried by the tracker on the same client.
use std::sync::{Arc, Mutex};

use cubecl::{bytes::Bytes, server::Handle};

#[derive(Default)]
enum Transfer {
    #[default]
    Empty,
    Pending {
        generation: u64,
        handles: Vec<Handle>,
    },
    Ready {
        generation: u64,
        bytes: Vec<Bytes>,
    },
}

#[derive(Default)]
struct Shared {
    generation: u64,
    transfer: Transfer,
}

/// Neither endpoint is cloneable; only the scanner can begin a generation.
pub(super) struct Producer(Arc<Mutex<Shared>>);
pub(super) struct Consumer(Arc<Mutex<Shared>>);

pub(super) fn endpoints() -> (Producer, Consumer) {
    let shared = Arc::new(Mutex::new(Shared::default()));
    (Producer(shared.clone()), Consumer(shared))
}

impl Producer {
    pub(super) fn abort(&self) {
        let mut shared = self.0.lock().unwrap_or_else(|error| error.into_inner());
        shared.generation = shared.generation.wrapping_add(1);
        shared.transfer = Transfer::Empty;
    }

    pub(super) fn stage(&self, handles: Vec<Handle>) {
        let mut shared = self.0.lock().unwrap_or_else(|error| error.into_inner());
        shared.transfer = Transfer::Pending {
            generation: shared.generation,
            handles,
        };
    }

    pub(super) fn take(&self) -> TransferResult {
        let mut shared = self.0.lock().unwrap_or_else(|error| error.into_inner());
        match std::mem::take(&mut shared.transfer) {
            Transfer::Ready { generation, bytes } if generation == shared.generation => {
                TransferResult::Ready(bytes)
            }
            Transfer::Pending {
                generation,
                handles,
            } if generation == shared.generation => TransferResult::Pending(handles),
            _ => TransferResult::Empty,
        }
    }
}

pub(super) enum TransferResult {
    Empty,
    Pending(Vec<Handle>),
    Ready(Vec<Bytes>),
}

impl Consumer {
    pub(super) fn take_staged(&self) -> Option<(u64, Vec<Handle>)> {
        let mut shared = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if let Transfer::Pending {
            generation,
            handles,
        } = &mut shared.transfer
        {
            Some((*generation, std::mem::take(handles)))
        } else {
            None
        }
    }

    pub(super) fn deliver(&self, generation: u64, bytes: Vec<Bytes>) {
        let mut shared = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if generation == shared.generation && matches!(shared.transfer, Transfer::Pending { .. }) {
            shared.transfer = Transfer::Ready { generation, bytes };
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;

    #[test]
    fn stale_delivery_cannot_replace_a_retry() {
        let (producer, consumer) = endpoints();
        producer.stage(Vec::new());
        let (old, _) = consumer.take_staged().expect("pending");
        producer.abort();
        producer.stage(Vec::new());
        consumer.deliver(old, Vec::new());
        assert!(matches!(producer.take(), TransferResult::Pending(_)));
    }

    #[test]
    fn delivery_is_spent_once() {
        let (producer, consumer) = endpoints();
        producer.stage(Vec::new());
        let (generation, _) = consumer.take_staged().expect("pending");
        consumer.deliver(generation, Vec::new());
        assert!(matches!(producer.take(), TransferResult::Ready(_)));
        assert!(matches!(producer.take(), TransferResult::Empty));
    }
}
