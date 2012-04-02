PROJECT=
LIB=-lcuda -lcudart
CFLAGS=-O2 -g -Wall -Werror

all: ${PROJECT}

${PROJECT}: ${PROJECT}.o ${PROJECT}.cc
	gcc -o ${PROJECT} ${CFLAGS} ${LIB} ${PROJECT}.c ${PROJECT}.o

${PROJECT}.o: ${PROJECT}.cu
	nvcc -c ${PROJECT}.cu

clean:
	rm ${PROJECT} *.o 
