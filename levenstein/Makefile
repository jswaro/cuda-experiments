PROJECT=levenshtein

all: ${PROJECT}

${PROJECT}: ${PROJECT}.o ${PROJECT}.cc
	nvcc -g -lcuda -lcudart -o ${PROJECT} ${PROJECT}.o ${PROJECT}.cc

${PROJECT}.o: ${PROJECT}.cu
	nvcc -c ${PROJECT}.cu

clean:
	rm ${PROJECT} *.o 
