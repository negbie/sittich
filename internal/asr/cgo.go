package asr

/*
#cgo LDFLAGS: -L${SRCDIR}/../../lib
#cgo LDFLAGS: -lsherpa-onnx-c-api -lsherpa-onnx-core -lsherpa-onnx-cxx-api -lsherpa-onnx-fst -lsherpa-onnx-fstfar -lsherpa-onnx-kaldifst-core -lkaldi-decoder-core -lkaldi-native-fbank-core -lssentencepiece_core -lkissfft-float -lonnxruntime -lsherpa-onnx-portaudio_static
#cgo LDFLAGS: -lstdc++ -lm -ldl -lpthread

#include <malloc.h>
static inline void do_malloc_trim() { malloc_trim(0); }
*/
import "C"

// trimCHeap tells the glibc allocator to return all freed memory pages back to
// the Linux kernel. This is necessary in Lazy Mode because deleting the ONNX
// recognizer frees ~2.6GB to the C heap, but glibc will hold those pages mapped
// (causing RSS to stay high) unless explicitly told to release them.
func trimCHeap() {
	C.do_malloc_trim()
}
