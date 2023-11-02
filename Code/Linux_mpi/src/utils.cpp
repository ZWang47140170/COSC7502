#include <vector>
#include <assert.h>
#include <cmath>
#include <random>
#include "utils.h"
#include <omp.h>

/**
 * A function to generate transmitting bit stream.
 * 
 * \param bit_stream The generated bit stream
 * \param N The number of bits
 */
void GenerateBitStream(std::vector<char>& bit_stream, int N)
{
	assert(N > 0);
	bit_stream.resize(N);
	#pragma omp parallel for
	for (int i = 0; i < bit_stream.size(); i++)
	{
		thread_local std::mt19937 generator(std::random_device{}());
		std::uniform_int_distribution<int> distribution(0, 1);
		bit_stream[i] = distribution(generator);
	}
}

/**
 * A function to finish modulation of BPSK.
 * 
 * \param bit_stream The bits to be modulated.
 * \param symbols The modulated symbols corresponding to bit stream.
 */
void BPSKModulation(std::vector<char>& bit_stream, std::vector<Complex>& symbols)
{
	assert(bit_stream.size() > 0);
	symbols.resize(bit_stream.size());
	#pragma omp parallel for
	for (int i = 0; i < symbols.size(); i++)
	{
		float phi = (float)bit_stream[i] * F_PI;
		symbols[i].x = cosf(phi);
		symbols[i].y = sinf(phi);
	}
}

/**
 * A function to simulate AWGN channel by adding Gaussion noise.
 * 
 * \param symbols The input and output buffer with input as clean BPSK symbols and output as BPSK symbols.
 * \param snr_db The signal to noise ratio in dB
 */
void AddNoise(std::vector<Complex>& symbols, float snr_db)
{
	assert(symbols.size() > 0);
	float noise_power= 0.5f / powf(10.0f, (snr_db / 10.0f));
	float sigma = sqrtf(noise_power);
	
	#pragma omp parallel for
	for (int i = 0; i < symbols.size(); i++)
	{
		thread_local std::mt19937 generator(std::random_device{}());
		std::normal_distribution<float> distribution(0.0f, sigma);
		symbols[i].x += distribution(generator);
		symbols[i].y += distribution(generator);
	}
}

/**
 * A function to convert BPSK symbols to bit stream.
 * 
 * \param symbols The BPSK symbols.
 * \param bit_stream_demod The demodulated bit stream.
 */
void BPSKDemodulation(std::vector<Complex>& symbols, std::vector<char>& bit_stream_demod)
{
	assert(symbols.size() > 0);
	float norm_factor = 2.0f / (2.0f * F_PI);
	bit_stream_demod.resize(symbols.size());
	#pragma omp parallel for
	for (int i = 0; i < bit_stream_demod.size(); i++)
	{
		float angle = atan2(symbols[i].y, symbols[i].x);
		bit_stream_demod[i] = (char)roundf(angle * norm_factor);
		bit_stream_demod[i] += bit_stream_demod[i] < 0 ? 2 : 0;
	}
}

/**
 * A function to calculate bit error rate.
 * 
 * \param bit_stream_demod The demodulated bit stream.
 * \param bit_stream_ref The reference bit stream.
 * \return 
 */
float CalculateBER(std::vector<char>& bit_stream_demod, std::vector<char>& bit_stream_ref)
{
	assert(bit_stream_demod.size() > 0 && bit_stream_demod.size() == bit_stream_ref.size());
	float N = (float)bit_stream_demod.size();
	float err_bit_cnt = 0.0f;
	#pragma omp parallel for reduction(+:err_bit_cnt)
	for (int i = 0; i < bit_stream_demod.size(); i++)
	{
		err_bit_cnt += bit_stream_demod[i] != bit_stream_ref[i] ? 1.0f : 0.0f;
	}
	return err_bit_cnt / N;
}