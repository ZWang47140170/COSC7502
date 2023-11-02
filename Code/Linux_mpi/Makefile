CC = mpicxx
CFLAGS = -Wall -lm -fopenmp

ber: opt_main.o CTime.o utils.o
	$(CC) $(CFLAGS) -o opt_ber opt_main.o CTime.o utils.o
 
opt_main.o: ./src/opt_main.cpp
	$(CC) $(CFLAGS) -c ./src/opt_main.cpp -o opt_main.o

CTime.o: ./src/CTime.cpp
	$(CC) $(CFLAGS) -c ./src/CTime.cpp -o CTime.o
 
utils.o: ./src/utils.cpp
	$(CC) $(CFLAGS) -c ./src/utils.cpp -o utils.o
 
clean:
	rm -rf *.o ber