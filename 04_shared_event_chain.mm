#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

static NSString* const kSrc = @R"(
#include <metal_stdlib>
using namespace metal;
// data[gid] に +X するだけの汎用カーネル
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

        // 追加値用の定数バッファ（SharedでOK）
        float v1 = 1.0f, v2 = 2.0f;
        id<MTLBuffer> cbuf1 = [dev newBufferWithBytes:&v1 length:sizeof(float)
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> cbuf2 = [dev newBufferWithBytes:&v2 length:sizeof(float)
                                              options:MTLResourceStorageModeShared];

        // 共有イベント（wait-value 相当）
        id<MTLSharedEvent> ev = [dev newSharedEvent];
        uint64_t signalValue = 1;

        // --- CB1: +1 して ev を signal(value=1)
        id<MTLCommandBuffer> cb1 = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc1 = [cb1 computeCommandEncoder];
        [enc1 setComputePipelineState:pso];
        [enc1 setBuffer:buf offset:0 atIndex:0];
        [enc1 setBuffer:cbuf1 offset:0 atIndex:1];
        MTLSize grid = MTLSizeMake(N,1,1);
        NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 64);
        [enc1 dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [enc1 endEncoding];
        [cb1 encodeSignalEvent:ev value:signalValue];

        // --- CB2: ev(value=1) を wait してから +2
        id<MTLCommandBuffer> cb2 = [q commandBuffer];
        [cb2 encodeWaitForEvent:ev value:signalValue];
        id<MTLComputeCommandEncoder> enc2 = [cb2 computeCommandEncoder];
        [enc2 setComputePipelineState:pso];
        [enc2 setBuffer:buf offset:0 atIndex:0];
        [enc2 setBuffer:cbuf2 offset:0 atIndex:1];
        [enc2 dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [enc2 endEncoding];

        // CPUは待たずに両方投げる（依存はGPU側イベントで担保）
        [cb1 commit];
        [cb2 commit];

        // 最終だけ確認のため待つ
        [cb2 waitUntilCompleted];

        std::cout << "Result: ";
        for (NSUInteger i=0;i<N;++i) std::cout << p[i] << " ";
        std::cout << "\n"; // 期待: (0..7) -> +1 -> +2 => 3..9
    }
    return 0;
}