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
#include <mpi.h> // include MPI header file

#define PERFORMANCE_PART

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get total number of processes

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

    if (rank == 0) std::cout << "Start simulation!\n";

#ifndef PERFORMANCE_PART
    timer.Tic();
#endif
    int i_start = rank * (snr_db_set.size() / size);
    int i_end = (rank == size - 1) ? snr_db_set.size() : (rank + 1) * (snr_db_set.size() / size);

    for (int i = i_start; i < i_end; i++)
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
    // gather results from all processes
    std::vector<float> total_elaps_gts(size), total_elaps_mod(size), total_elaps_awgn(size), total_elaps_demod(size), total_elaps_ber(size);
    MPI_Gather(&elaps_gts, 1, MPI_FLOAT, total_elaps_gts.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&elaps_mod, 1, MPI_FLOAT, total_elaps_mod.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&elaps_awgn, 1, MPI_FLOAT, total_elaps_awgn.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&elaps_demod, 1, MPI_FLOAT, total_elaps_demod.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&elaps_ber, 1, MPI_FLOAT, total_elaps_ber.data(), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float avg_elaps_gts = std::accumulate(total_elaps_gts.begin(), total_elaps_gts.end(), 0.0f) / snr_db_set.size();
        float avg_elaps_mod = std::accumulate(total_elaps_mod.begin(), total_elaps_mod.end(), 0.0f) / snr_db_set.size();
        float avg_elaps_awgn = std::accumulate(total_elaps_awgn.begin(), total_elaps_awgn.end(), 0.0f) / snr_db_set.size();
        float avg_elaps_demod = std::accumulate(total_elaps_demod.begin(), total_elaps_demod.end(), 0.0f) / snr_db_set.size();
        float avg_elaps_ber = std::accumulate(total_elaps_ber.begin(), total_elaps_ber.end(), 0.0f) / snr_db_set.size();

        printf("Runtime of GenerateBitStream: %.3f ms\n", avg_elaps_gts);
        printf("Runtime of BPSKModulation: %.3f ms\n", avg_elaps_mod);
        printf("Runtime of AddNoise: %.3f ms\n", avg_elaps_awgn);
        printf("Runtime of BPSKDemodulation: %.3f ms\n", avg_elaps_demod);
        printf("Runtime of CalculateBER: %.3f ms\n", avg_elaps_ber);
    }
#endif

    // gather results from all processes
    std::vector<float> total_ber(snr_db_set.size());
    MPI_Gather(ber.data() + i_start, i_end - i_start, MPI_FLOAT, total_ber.data(), i_end - i_start, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        // save results
        FILE* fp = fopen("result.txt","w");
        for (int i = 0; i < snr_db_set.size(); i++)
        {
            fprintf(fp, "%f %f\n", snr_db_set[i], total_ber[i]);
        }
        fclose(fp);
        std::cout << "Simulation done!\n";
    }

    MPI_Finalize(); // finalize MPI
    return 0;
}