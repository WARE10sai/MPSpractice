#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::cerr << "No Metal device\n"; return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSLog(@"Device: %@", dev.name);
        std::cout << "Queue created\n";
        // [q release];
    }
    return 0;
}