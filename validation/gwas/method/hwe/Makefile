all: clean chwe.so

clean:
	rm -f *.o *.so 

chwe.so: chwe.o
	gcc -shared -o libchwe.so chwe.o

chwe.o: chwe.c
	gcc -c -Wall -Werror -fpic chwe.c


