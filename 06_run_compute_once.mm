#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>

// ====== Buffer→値渡し 自動バインド（既存ルール踏襲） ======
static inline void bind_buffers_then_vals(id<MTLComputeCommandEncoder> enc,
                                          const std::vector<id<MTLBuffer>>& bufs,
                                          const std::vector<std::pair<const void*, size_t>>& vals) {
    uint32_t idx = 0;
    for (auto b : bufs) { [enc setBuffer:b offset:0 atIndex:idx++]; }
    for (auto& v : vals) { [enc setBytes:v.first length:v.second atIndex:idx++]; }
}

// ====== 1回分の計算をまるごと実行（非キャッシュ版） ======
static bool run_compute_once(
    id<MTLDevice> dev,
    id<MTLCommandQueue> q,
    id<MTLLibrary> lib,
    NSString* fnName,
    const std::vector<id<MTLBuffer>>& bufs,
    const std::vector<std::pair<const void*, size_t>>& vals,
    MTLSize grid,
    MTLSize tg,
    bool wait_cpu = true,
    NSError** err_out = nullptr)
{
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:fnName];
    if (!fn) { if (err_out) *err_out = err; return false; }

    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { if (err_out) *err_out = err; return false; }

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];

    bind_buffers_then_vals(enc, bufs, vals);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cb commit];
    if (wait_cpu) [cb waitUntilCompleted];
    return true;
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

        const NSUInteger N = 8;
        id<MTLBuffer> buf = [dev newBufferWithLength:N*sizeof(float)
                                             options:MTLResourceStorageModeShared];
        float* p = (float*)buf.contents;
        for (NSUInteger i=0;i<N;++i) p[i] = (float)i; // 0..7

        float v1 = 1.0f, v2 = 2.0f;

        // tg（blockDim 相当）はPSOがなくても仮に64でOK（最適化は後で）
        MTLSize grid = MTLSizeMake(N,1,1);
        MTLSize tgroup = MTLSizeMake(64,1,1);

        // --- 1回目：+1
        if (!run_compute_once(dev, q, lib, @"add_val",
                              /*buffers*/ { buf },
                              /*vals*/    { {&v1, sizeof(v1)} },
                              grid, tgroup, /*wait_cpu=*/true, &err)) {
            NSLog(@"Run1 error: %@", err); return 1;
        }

        // --- 2回目：+2
        if (!run_compute_once(dev, q, lib, @"add_val",
                              /*buffers*/ { buf },
                              /*vals*/    { {&v2, sizeof(v2)} },
                              grid, tgroup, /*wait_cpu=*/true, &err)) {
            NSLog(@"Run2 error: %@", err); return 1;
        }

        std::cout << "Result: ";
        for (NSUInteger i=0;i<N;++i) std::cout << p[i] << " ";
        std::cout << "\n"; // 期待: 3..10
    }
    return 0;
}