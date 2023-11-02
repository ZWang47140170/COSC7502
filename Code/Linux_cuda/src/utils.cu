#include <vector>
#include <assert.h>
#include <cmath>
#include <random>
#include <cuda.h>
#include <curand_kernel.h>
#include "utils.h"

// CUDA kernel for GenerateBitStream
__global__ void GenerateBitStreamKernel(char* d_bit_stream, int N_bits) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N_bits) {
        curandState state;
        curand_init((unsigned long long)clock() + idx, 0, 0, &state);
        d_bit_stream[idx] = (curand_uniform(&state) > 0.5) ? 1 : 0;
    }
}

void GenerateBitStream(char* d_bit_stream, int N_bits) {
    int blockSize = 256;
    int gridSize = (N_bits + blockSize - 1) / blockSize;
    GenerateBitStreamKernel<<<gridSize, blockSize>>>(d_bit_stream, N_bits);
}

__global__ void BPSKModulationKernel(char* d_bit_stream, Complex* d_symbols, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_symbols[idx].x = (d_bit_stream[idx] == 0) ? -1.0f : 1.0f;
        d_symbols[idx].y = 0;
    }
}

void BPSKModulation(char* d_bit_stream, Complex* d_symbols, int N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    BPSKModulationKernel<<<gridSize, blockSize>>>(d_bit_stream, d_symbols, N);
}

__global__ void AddNoiseKernel(Complex* d_symbols, float snr, int N) {
    // implement your kernel function here
}

void AddNoise(Complex* d_symbols, float snr, int N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    AddNoiseKernel<<<gridSize, blockSize>>>(d_symbols, snr, N);
}

__global__ void BPSKDemodulationKernel(Complex* d_symbols, char* d_bit_stream_demod, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_bit_stream_demod[idx] = (d_symbols[idx].x > 0) ? 1 : 0;
    }
}

void BPSKDemodulation(Complex* d_symbols, char* d_bit_stream_demod, int N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    BPSKDemodulationKernel<<<gridSize, blockSize>>>(d_symbols, d_bit_stream_demod, N);
}

__global__ void CalculateBERKernel(char* d_bit_stream_demod, char* d_bit_stream_ref, float* d_ber, int N_bits) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N_bits) {
        atomicAdd(d_ber, (d_bit_stream_demod[idx] != d_bit_stream_ref[idx]) ? 1 : 0);
    }
}

float CalculateBER(char* d_bit_stream_demod, char* d_bit_stream_ref, int N_bits) {
    float* d_ber;
    cudaMalloc(&d_ber, sizeof(float));
    float h_ber = 0.0f;
    cudaMemcpy(d_ber, &h_ber, sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (N_bits + blockSize - 1) / blockSize;
    CalculateBERKernel<<<gridSize, blockSize>>>(d_bit_stream_demod, d_bit_stream_ref, d_ber, N_bits);
    
    cudaMemcpy(&h_ber, d_ber, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_ber);
    
    return h_ber / N_bits;
}