PROJECT=matrix
LIB=-lcuda
#LIB=-lcuda -lcudart
CFLAGS=-O2 -g -Wall -Werror

all: ${PROJECT}

${PROJECT}: ${PROJECT}.o ${PROJECT}.cc
	nvcc -o ${PROJECT} ${PROJECT}.o ${PROJECT}.cc

${PROJECT}.o: ${PROJECT}.cu
	nvcc -c ${PROJECT}.cu

clean:
	rm ${PROJECT} *.o 
