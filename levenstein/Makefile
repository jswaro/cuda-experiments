PROJECT=levenshtein
NVCCFLAGS=-arch=sm_11
all: ${PROJECT}

${PROJECT}: ${PROJECT}.o ${PROJECT}.cc
	nvcc ${NVCCFLAGS} -g -lcuda -lcudart -o ${PROJECT} ${PROJECT}.o ${PROJECT}.cc

${PROJECT}.o: ${PROJECT}.cu
	nvcc ${NVCCFLAGS} -g -c ${PROJECT}.cu

clean:
	rm ${PROJECT} *.o 
