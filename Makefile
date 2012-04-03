PROJECT=

all: ${PROJECT}

${PROJECT}: ${PROJECT}.o ${PROJECT}.cc
	nvcc -o ${PROJECT} ${PROJECT}.o ${PROJECT}.cc

${PROJECT}.o: ${PROJECT}.cu
	nvcc -c ${PROJECT}.cu

clean:
	rm ${PROJECT} *.o 
