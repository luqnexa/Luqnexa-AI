// SPDX-License-Identifier: AGPL-3.0-only
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Counted;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
// This separate test executable has one test and only counts around inference.
unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counted = Counted;

#[test]
fn repeated_unicode_inference_has_no_heap_allocations_after_loading() {
    let model = luqnexa_ml::Trainer::new(1337).quantize();
    let input = "Pythonでモデルを検証する";
    let context = "直前の文脈：日本語とEnglishの混在入力";
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..100 {
        let prepared = model.prepare(std::hint::black_box(input), context);
        for surface in ["Pythonでモデルを検証する", "Pythonでモデルを検証しない"]
        {
            std::hint::black_box(model.score(&prepared, surface));
        }
    }
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    assert_eq!(before, after);
}
