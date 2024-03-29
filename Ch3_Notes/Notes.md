# Chapter Objectives

* Write your first lines of code in CUDA C
* Learn the difference between code written for the host and code written for a device
* Lean about the ways device memory can be used on CUDA-capable devices
* Learn how to query your system for information on its CUDA-capable devices


# Sections
3.2 A First Program  
3.3 Querying Devices   
3.4 Using Device Properties    
3.5 Chapter Review


# 3.2:

## First Program - Hello World

```c
#include "../common/book.h"
int main( void ) {
printf( "Hello, World!\n" );
return 0;
}
```

## A kernel call
```c
#include <iostream>

__global__void kernel ( void ) {
}

int main( void ) {
  kernel<<1,1>>>();
  printf( "Hello, World!\n" );
  return 0;
}
```

### Differences between first program:
* An empty function named kernel() qualified with __global__
* A call to the empty function, embellished with <<<1,1>>>

__global__ tells the compiler that a function should be compiled to run on a device instead of host.  
So nvcc would give the function lkernel() to the compiler that handles device code and give main() to host compiler  
The angle brackets denote arguments planned to pass to the runtime system, influence how the runtime will launch our device code  
The arguements inside the parentheses get passed to the device code

## Passing Parameters

```c
#include <iostream>
#include "book.h"
__global__ void add( int a, int b, int *c ) {
*c = a + b;
}
int main( void ) {
int c;
int *dev_c;
HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
add<<<1,1>>>( 2, 7, dev_c );
HANDLE_ERROR( cudaMemcpy( &c,
dev_c,
sizeof(int),
cudaMemcpyDeviceToHost ) );
printf( "2 + 7 = %d\n", c );
cudaFree( dev_c );
return 0;
}
```

### Concepts introduced:
* Can pass parameters to a kernel as we would with any C function
* Need to allocate memory to do anything useful on a device, such as return values to the host

Do not dereference the pointer returned by cudaMalloc() from code that executes on the host. Host code may pass the pointer around, perform arithmetic on it , or even cast it to a different type. but you cannot ue it to read or write from memory.

However, the compiler does not recognize if you've made a mistake. will allow deferences of device pointers in your host code since it looks like any other pointer in the app.

### Restirctions on the usage of device pointer:
* Can pass pointers allocated with cudamalloc() to functions that execute on the device
* Can use pointers allocated with cudaMalloc() to read or write memory from code that executes on the device.
* Can pass pointers allocated with cudaMalloc() to functions that execute on the host.
* Cannot use pointers allocated with cudaMalloc() to read or write memory from code that executes on the host.

free() function can not be used to release memoryallocated with cudaMalloc().
Have to use cudaFree()

This code shows that device memory can be accessed by using device pointers from within device code and by using calls to cudaMemcpy()

so in the line *c = a + b we are adding the parameters a and b together and storing the result in the memory pointed to by c.

The restrictions that we see for device pointers also holds true for host pointers. host pointers can access memory from host code. device pointers can access memory from device code.

## Querying Devices
Need to know how much memory and what types of capabilities available CUDA devices have. And to destinguish which you are using when there is more than one available.

```c
int count;
HANDLE_ERROR( cudaGetDeviceCount( &count ) );


struct cudaDeviceProp {
char name[256];
size_t totalGlobalMem;
size_t sharedMemPerBlock;
int regsPerBlock;
int warpSize;
size_t memPitch;
int maxThreadsPerBlock;
int maxThreadsDim[3];
int maxGridSize[3];
size_t totalConstMem;
int major;
int minor;
int clockRate;
size_t textureAlignment;
int deviceOverlap;
int multiProcessorCount;
int kernelExecTimeoutEnabled;
int integrated;
int canMapHostMemory;
int computeMode;
int maxTexture1D;
int maxTexture2D[2];
int maxTexture3D[3];
int maxTexture2DArray[3];
int concurrentKernels;
}
```

|DEvICE ProPErty | Description |
|---|---|
|`char name[256];` | An ASCII string identifying the device (e.g., "GeForce GTX 280")|
|`size_t totalGlobalMem` | The amount of global memory on the device in bytes|
|`size_t sharedMemPerBlock` | The maximum amount of shared memory a single block may use in bytes|
|`int regsPerBlock` | The number of 32-bit registers available per block|
|`int warpSize` | The number of threads in a warp|
|`size_t memPitch` | The maximum pitch allowed for memory copies in bytes|
|`int maxThreadsPerBlock` | The maximum number of threads that a block may contain|
|`int maxThreadsDim[3]` |The maximum number of threads allowed along each dimension of a block|
|`int maxGridSize[3]` |The number of blocks allowed along each dimension of a grid|
|`size_t totalConstMem` |The amount of available constant memory|
|`int major` | The major revision of the device’s compute capability|
|`int minor` |The minor revision of the device’s compute capability|
|`size_t textureAlignment` |The device’s requirement for texture alignment|
|`int deviceOverlap` | A boolean value representing whether the device can simultaneously perform a cudaMemcpy() and kernel execution|
|`int multiProcessorCount` |The number of multiprocessors on the device|
|`int kernelExecTimeoutEnabled` | A boolean value representing whether there is a runtime limit for kernels executed on this device|
|`int integrated` |A boolean value representing whether the device is an integrated GPU (i.e., part of the chipset and not a discrete GPU)|
|`int canMapHostMemory` | A boolean value representing whether the device can map host memory into the CUDA device address space|
|`int computeMode` |A value representing the device’s computing mode: default, exclusive, or prohibited|
|`int maxTexture1D` |The maximum size supported for 1D textures|
|`int maxTexture2D[2]` |The maximum dimensions supported for 2D textures|
|`int maxTexture3D[3]` | The maximum dimensions supported for 3D textures|
|`int maxTexture2DArray[3]` |The maximum dimensions supported for 2D texture arrays|
|`int concurrentKernels` |A boolean value representing whether the device supports executing multiple kernels within the same context simultaneously|

Consult the NVIDIA CUDA Programming Guide for more information on the important details of these properties

```c
#include "../common/book.h"
int main( void ) {
cudaDeviceProp prop;
int count;
HANDLE_ERROR( cudaGetDeviceCount( &count ) );
for (int i=0; i< count; i++) {
HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
//Do something with our device's properties
}
}
```

This code is the beginning of our device query. 

```c
#include "../common/book.h"
int main( void ) {
cudaDeviceProp prop;
int count;
HANDLE_ERROR( cudaGetDeviceCount( &count ) );
for (int i=0; i< count; i++) {
HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
printf( " --- General Information for device %d ---\n", i );
printf( "Name: %s\n", prop.name );
printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
printf( "Clock rate: %d\n", prop.clockRate );
printf( "Device copy overlap: " );
if (prop.deviceOverlap)
printf( "Enabled\n" );
else
printf( "Disabled\n" );
printf( "Kernel execition timeout : " );
if (prop.kernelExecTimeoutEnabled)
printf( "Enabled\n" );
else
printf( "Disabled\n" );
printf( " --- Memory Information for device %d ---\n", i );
printf( "Total global mem: %ld\n", prop.totalGlobalMem );
printf( "Total constant Mem: %ld\n", prop.totalConstMem );
printf( "Max mem pitch: %ld\n", prop.memPitch );
printf( "Texture Alignment: %ld\n", prop.textureAlignment );
printf( " --- MP Information for device %d ---\n", i );
printf( "Multiprocessor count: %d\n",
prop.multiProcessorCount );
printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
printf( "Registers per mp: %d\n", prop.regsPerBlock );
printf( "Threads in warp: %d\n", prop.warpSize );
printf( "Max threads per block: %d\n",
prop.maxThreadsPerBlock );
printf( "Max thread dimensions: (%d, %d, %d)\n",
prop.maxThreadsDim[0], prop.maxThreadsDim[1],
prop.maxThreadsDim[2] );
printf( "Max grid dimensions: (%d, %d, %d)\n",
prop.maxGridSize[0], prop.maxGridSize[1],
prop.maxGridSize[2] );
printf( "\n" );
}
}
```

Expand code to print the device properties

## Using Device Properties
Why would be be interested in device properties?

1) We would want to run our software on specific GPUs such as one with the  most multiprocessors or dprcific compute capabilities.
2) If the kernel needs close interaction with the CPU, we use the integrated GPU that shares system memory with CPU.

Would utilize cudaGetDeviceProperties()

or need double-precision floating -point support, we would need a device with compute capabilitiy of 1.3 or higher.

```c
cudaDeviceProp prop;
memset( &prop, 0, sizeof( cudaDeviceProp ) );
prop.major = 1;
prop.minor = 3;
```

This piece of code fills the cudaDeviceProp structure with the properties that we need the device to have.

Then it has to be passed to cudaChooseDevice() to have the CUDA runtime find a device that satisfies these constraints.\
The call cudaChooseDevice() returns a device ID. We then pass to cudaSetDevice() which sets our device and from there all device operations will take place on that device found in cudaChooseDevice().

```c
#include "../common/book.h"
int main( void ) {
cudaDeviceProp prop;
int dev;
HANDLE_ERROR( cudaGetDevice( &dev ) );
printf( "ID of current CUDA device: %d\n", dev );
memset( &prop, 0, sizeof( cudaDeviceProp ) );
prop.major = 1;
prop.minor = 3;
HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
printf( "ID of CUDA device closest to revision 1.3: %d\n", dev );
HANDLE_ERROR( cudaSetDevice( dev ) );
}
```

Since our application may depend on certain features of the GPU or needs the fastest GPU or a GPU with an higher affinity to the host then we need to be familiar with this APU since ther is no guarantee that the CUDA runtime will choose the best or most appropriate GPU for our app.


## Key Terms:
* host: CPU and its system memory
* device: GPU and its memory
* kernel: function that executes on the device
* cudaMalloc(): allocation of memory. Tells CUDA runtime to allocate the memory on the device
* pointer: Holds the address of the newly allocated memory
* HANDLE_ERROR(): utility macro that detects that the call has returned an error, prints the associated error message, and exits the application with an EXIT_FAILURE code. (Highly likely to be insufficient in production code)
* cudaMemcpy(): call to access memory on a device.
* cudaMemcpyDeviceToHost: parameter in cudaMemcpy() instructing runtime that the source pointer is a device pointer and the destinationpointer is a host pointer.
* cudaMemcpyHostToDevice: parameter in cudaMemcpy() instructing runtime that the source data is on the host and the destination is an address on the device.
* cudaMemcpyDeviceToDevice: specifies that both pointers are on the device
* memcpy(): would be used if bother pointers were on host
* cudaGetDeviceCount(): call to get the count of CUDA devices
* cudaDeviceProp: structure type returned when cudaGetDeviceCOunt() is called
