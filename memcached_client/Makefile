
all: *.c *.h
	gcc -c *.c -Wall -levent  -pthread -lm -D_GNU_SOURCE
	g++ -c cpp/Data.cpp -o Data.o
	g++ -c cpp/Matrix.cpp -o Matrix.o
	g++ -c cpp/ARmodel.cpp -o ARmodel.o
	g++ -o loader *.o -Wall -levent  -pthread -lm -D_GNU_SOURCE
	#gcc -O3 *.c -Wall -levent  -pthread -lm -D_GNU_SOURCE -o loader
clean:
	rm loader *.o
