# Chapter Objectives

* Learn one of the fundamental ways CUDA exposes its parallelism
* Write your first parallel code with CUDA C


# Sections
4.2 CUDA Parallel Programming  
4.3 Chapter Review

## Summing Vectors
This example should illustrate threads and how we use them to code with CUDA C

Linear algebra style computation of summing two vectors:  
Lets say we have 2 lists of numbers where we want to sum corresponding elements of each list and store the result in a third list  

![image](https://github.com/Q-SKADOO/Cuda_By_Example/assets/112571800/22b5d0c7-acac-41b6-b90b-0335c751396a)

### CPU Vector Sums
Traditional C code for vector summation  

```c
#include "../common/book.h"
#define N 10
void add( int *a, int *b, int *c ) {
int tid = 0; // this is CPU zero, so we start at zero
while (tid < N) {
c[tid] = a[tid] + b[tid];
tid += 1; // we have one CPU, so we increment by one
}
}
int main( void ) {
int a[N], b[N], c[N];
// fill the arrays 'a' and 'b' on the CPU
for (int i=0; i<N; i++) {
a[i] = -i;
b[i] = i * i;
}
add( a, b, c );
// display the results
for (int i=0; i<N; i++) {
printf( "%d + %d = %d\n", a[i], b[i], c[i] );
}
return 0;
}
```

Let's take a look specifically at the add() function and why it is over complicated in the book.

```c
void add( int *a, int *b, int *c ) {
int tid = 0; // this is CPU zero, so we start at zero
while (tid < N) {
c[tid] = a[tid] + b[tid];
tid += 1; // we have one CPU, so we increment by one
}
}
```

In the code above, the sum is computed with a while loop, where the index `tid` ranges from `0` to `N-1`.  
The corresponding elements of `a[]` and `b[]` are added, placing the result in the corresponding element of `c[]`  

This code would typically be written in a simpler form, such as below:

```c
void add( int *a, int *b, int *c ) {
for (i=0; i < N; i++) {
c[i] = a[i] + b[i];
}
}
```

The slightly more convoluted version was shown to suggest a potential way to parallelize code on a system with multiple CPUs or CPU cores.  
For example, with a dual-core processor, one could change the increment to 2 and have one core initialize the loop with `tid = 0` and another with `tid = 1`.  
The 1st core adds up the even-indexed elements while the second core adds the odd-indexed elements. 

This amounts to executing the following code on each of the two CPU cores:  

|CPU Core 1 | CPU Core 2|
|---|---|
|```c
void add( int *a, int *b, int *c )
{
int tid = 0;
while (tid < N) {
c[tid] = a[tid] + b[tid];
tid += 2;
}
}
```| ```c
void add( int *a, int *b, int *c )
{
int tid = 1;
while (tid < N) {
c[tid] = a[tid] + b[tid];
tid += 2;
}
}
```|




