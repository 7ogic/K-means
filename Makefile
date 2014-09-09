
CXX=g++
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq opencl opencl2

.PHONY: all seq opencl opencl2 clean


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}

opencl: kmeans_opencl

kmeans_opencl: kmeans_opencl.o kmeans_main.o cl_util.o
	${CXX} $^ -o $@ ${LDFLAGS} -lOpenCL

opencl2: kmeans_opencl_2

kmeans_opencl_2: kmeans_opencl_2.o kmeans_main.o cl_util.o
	${CXX} $^ -o $@ ${LDFLAGS} -lOpenCL

clean:
	rm -f kmeans kmeans_main.o kmeans_seq.o kmeans_opencl.o cl_util.o kmeans_opencl_2.o
