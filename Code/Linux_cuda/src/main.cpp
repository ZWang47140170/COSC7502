#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif
#include <stdio.h>
#include "utils.h"
#include "CTime.h"

#define PERFORMANCE_PART

int main()
{
    CTime timer;// performance timer

    // generate SNR set
    float max_snr_db = 10.0f;
    std::vector<float> snr_db_set;
    float snr_tmp = -5.0f;
    float snr_step = 0.5f;
    while (snr_tmp < max_snr_db)
    {
        snr_db_set.push_back(snr_tmp);
        snr_tmp += snr_step;
    }

    int N_bits = 10000;// length of bit stream
    std::vector<char> bit_stream_ref;
    std::vector<Complex> symbols;
    std::vector<char> bit_stream_demod;
    std::vector<float> ber(snr_db_set.size());

    // vVariables for recording duration
    float elaps_gts = 0.0f;
    float elaps_mod = 0.0f;
    float elaps_awgn = 0.0f;
    float elaps_demod = 0.0f;
    float elaps_ber = 0.0f;
    std::cout << "Start simulation!\n";
#ifndef PERFORMANCE_PART
    timer.Tic();
#endif
    for (int i = 0; i < snr_db_set.size(); i++)
    {
#ifdef PERFORMANCE_PART
        timer.Tic();
#endif
        // generate bit stream
        GenerateBitStream(bit_stream_ref, N_bits);
#ifdef PERFORMANCE_PART
        timer.Toc();
        elaps_gts += timer.GetElaps();
#endif

#ifdef PERFORMANCE_PART
        timer.Tic();
#endif
        // convert bit stream to BPSK symbols
        BPSKModulation(bit_stream_ref, symbols);
#ifdef PERFORMANCE_PART
        timer.Toc();
        elaps_mod += timer.GetElaps();
#endif

#ifdef PERFORMANCE_PART
        timer.Tic();
#endif
        // simulate AWGN channel
        AddNoise(symbols, snr_db_set[i]);
#ifdef PERFORMANCE_PART
        timer.Toc();
        elaps_awgn += timer.GetElaps();
#endif

#ifdef PERFORMANCE_PART
        timer.Tic();
#endif
        // convert BPSK symbols to bit stream
        BPSKDemodulation(symbols, bit_stream_demod);
#ifdef PERFORMANCE_PART
        timer.Toc();
        elaps_demod += timer.GetElaps();
#endif

#ifdef PERFORMANCE_PART
        timer.Tic();
#endif
        // calculate bit error rate
        ber[i] = CalculateBER(bit_stream_demod, bit_stream_ref);
#ifdef PERFORMANCE_PART
        timer.Toc();
        elaps_ber += timer.GetElaps();
#endif
    }
#ifndef PERFORMANCE_PART
    timer.Toc("Total processing");
#else
    printf("Runtime of GenerateBitStream: %.3f ms\n", elaps_gts / snr_db_set.size());
    printf("Runtime of BPSKModulation: %.3f ms\n", elaps_mod / snr_db_set.size());
    printf("Runtime of AddNoise: %.3f ms\n", elaps_awgn / snr_db_set.size());
    printf("Runtime of BPSKDemodulation: %.3f ms\n", elaps_demod / snr_db_set.size());
    printf("Runtime of CalculateBER: %.3f ms\n", elaps_ber / snr_db_set.size());
#endif
    //FILE* fp0 = fopen("symbol.txt", "w");
    //for (int i = 0; i < symbols.size(); i++)
    //{
    //    fprintf(fp0, "%d %f %f\n", bit_stream_ref[i], symbols[i].x, symbols[i].y);
    //}
    //fclose(fp0);

    // save results
    FILE* fp = fopen("result.txt","w");
    for (int i = 0; i < snr_db_set.size(); i++)
    {
        fprintf(fp, "%f %f\n", snr_db_set[i], ber[i]);
    }
    fclose(fp);
    std::cout << "Simulation done!\n";
}