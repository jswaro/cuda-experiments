PROJECT=hello_world
LIB=-lcuda -lcudart
CFLAGS=-O2 -g -Wall -Werror

all: ${PROJECT}

${PROJECT}: ${PROJECT}.o
	gcc -o ${PROJECT} ${CFLAGS} ${LIB} ${PROJECT}.o

${PROJECT}.o: ${PROJECT}.cu
	nvcc -c ${PROJECT}.cu

clean:
	rm ${PROJECT} *.o 
