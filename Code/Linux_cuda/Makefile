NVCC = nvcc
NVCCFLAGS = -arch=sm_86

ber: opt_main.o CTime.o utils.o
	$(NVCC) $(NVCCFLAGS) -o opt_ber_cuda opt_main.o CTime.o utils.o -lcudart

opt_main.o: ./src/opt_main.cu
	$(NVCC) $(NVCCFLAGS) -c ./src/opt_main.cu -o opt_main.o

CTime.o: ./src/CTime.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c ./src/CTime.cpp -o CTime.o

utils.o: ./src/utils.cu
	$(NVCC) $(NVCCFLAGS) -c ./src/utils.cu -o utils.o

clean:
	rm -rf *.o ber