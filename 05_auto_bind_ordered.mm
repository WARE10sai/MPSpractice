#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>

// ====== ここが「自動バインド：Buffer→値渡し」本体 ======
static inline void bind_buffers_then_vals(id<MTLComputeCommandEncoder> enc,
                                          const std::vector<id<MTLBuffer>>& bufs,
                                          const std::vector<std::pair<const void*, size_t>>& vals) {
    uint32_t idx = 0;
    // 1) まず MTLBuffer を 0.. に順番バインド（offset=0）
    for (auto b : bufs) {
        [enc setBuffer:b offset:0 atIndex:idx++];
    }
    // 2) 次に 値渡し（setBytes）を続けてバインド
    for (auto& v : vals) {
        [enc setBytes:v.first length:v.second atIndex:idx++];
    }
}

// ====== カーネル（値渡し val を読み取り専用で受ける） ======
static NSString* const kSrc = @R"(
#include <metal_stdlib>
using namespace metal;
kernel void add_val(device float* data [[buffer(0)]],
                    constant float& val [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    data[gid] += val;
}
)";

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil error:&err];
        if (!lib) { NSLog(@"Lib error: %@", err); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"add_val"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { NSLog(@"PSO error: %@", err); return 1; }

        const NSUInteger N = 8;
        id<MTLBuffer> buf = [dev newBufferWithLength:N*sizeof(float)
                                             options:MTLResourceStorageModeShared];
        float* p = (float*)buf.contents;
        for (NSUInteger i=0;i<N;++i) p[i] = (float)i; // 0..7

        float v1 = 1.0f, v2 = 2.0f;

        // --- CB1: +v1
        id<MTLCommandBuffer> cb1 = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc1 = [cb1 computeCommandEncoder];
        [enc1 setComputePipelineState:pso];

        bind_buffers_then_vals(enc1,
            /*buffers*/ { buf },
            /*vals*/    { {&v1, sizeof(v1)} }  // 値渡しはここに並べる
        );

        MTLSize grid = MTLSizeMake(N,1,1);
        NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 64);
        [enc1 dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [enc1 endEncoding];
        [cb1 commit];
        [cb1 waitUntilCompleted];

        // --- CB2: +v2
        id<MTLCommandBuffer> cb2 = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
        [enc2 setComputePipelineState:pso];

        bind_buffers_then_vals(enc2,
            /*buffers*/ { buf },                  // [[buffer(0)]]
            /*vals*/    { {&v2, sizeof(v2)} }     // [[buffer(1)]]
        );

        [enc2 dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [enc2 endEncoding];
        [cb2 commit];
        [cb2 waitUntilCompleted];

        std::cout << "Result: ";
        for (NSUInteger i=0;i<N;++i) std::cout << p[i] << " ";
        std::cout << "\n"; // 期待: (0..7) -> +1 -> +2 => 3..10
    }
    return 0;
}