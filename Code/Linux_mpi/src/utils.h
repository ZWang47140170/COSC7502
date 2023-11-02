#pragma once

#define F_PI 3.14159265358979f

typedef struct Complex_
{
	float x;
	float y;
}Complex;

void GenerateBitStream(std::vector<char>& bit_stream, int N);
void BPSKModulation(std::vector<char>& bit_stream, std::vector<Complex>& symbols);
void AddNoise(std::vector<Complex>& symbols, float snr);
void BPSKDemodulation(std::vector<Complex>& symbols, std::vector<char>& bit_stream_demod);
float CalculateBER(std::vector<char>& bit_stream_demod, std::vector<char>& bit_stream_ref);
