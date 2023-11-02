#pragma once

#define F_PI 3.14159265358979f

typedef struct Complex_
{
	float x;
	float y;
}Complex;

void GenerateBitStream(char* d_bit_stream, int N);
void BPSKModulation(char* d_bit_stream, Complex* d_symbols, int N);
void AddNoise(Complex* d_symbols, float snr, int N);
void BPSKDemodulation(Complex* d_symbols, char* d_bit_stream_demod, int N);
float CalculateBER(char* d_bit_stream_demod, char* d_bit_stream_ref, int N);