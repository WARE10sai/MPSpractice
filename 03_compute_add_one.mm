#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

static NSString* const kSrc = @R"(
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* data [[buffer(0)]],
                    uint gid [[thread_position_in_grid]]) {
    data[gid] += 1.0f;
}
)";

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil error:&err];
        if (!lib) { NSLog(@"Lib error: %@", err); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"add_one"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { NSLog(@"PSO error: %@", err); return 1; }

        const NSUInteger N = 16;
        id<MTLBuffer> buf = [dev newBufferWithLength:N*sizeof(float)
                                             options:MTLResourceStorageModeShared];
        float* p = (float*)buf.contents;
        for (NSUInteger i=0;i<N;++i) p[i] = (float)i;

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:buf offset:0 atIndex:0];

        MTLSize grid    = MTLSizeMake(N, 1, 1);
        NSUInteger tgSz = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 64);
        MTLSize tgroup  = MTLSizeMake(tgSz, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:tgroup];
        [enc endEncoding];

        [cb commit];
        [cb waitUntilCompleted]; // まずは同期で確認

        std::cout << "After: ";
        for (NSUInteger i=0;i<N;++i) std::cout << p[i] << " ";
        std::cout << "\n";
    }
    return 0;
}