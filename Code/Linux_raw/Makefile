CC = g++
CFLAGS = -Wall 
LDFLAGS = -lm

ber: main.o CTime.o utils.o
	$(CC) $(CFLAGS) -o ber main.o CTime.o utils.o
 
main.o: ./src/main.cpp
	$(CC) $(CFLAGS) -c ./src/main.cpp -o main.o
 
CTime.o: ./src/CTime.cpp
	$(CC) $(CFLAGS) -c ./src/CTime.cpp -o CTime.o
 
utils.o: ./src/utils.cpp
	$(CC) $(CFLAGS) -c ./src/utils.cpp -o utils.o
 
clean:
	rm -rf *.o ber