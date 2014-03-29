# Makefile for systemSolve example

include ../common/common.mk

INCLUDES=-I${CULA_INC_PATH}
LIBPATH32=-L${CULA_LIB_PATH_32}
LIBPATH64=-L${CULA_LIB_PATH_64}

LIBS=-lcula_lapack -lcublas -lcudart -liomp5

usage:
	@echo "To build this example, type one of:"
	@echo ""
	@echo "    make build32"
	@echo "    make build64"
	@echo "    make build64gpu"
	@echo ""
	@echo "where '32' and '64' represent the platform you wish to build for"

build32:
	sh ../checkenvironment.sh
	${CC} -m32 -o systemSolve systemSolve_new.c $(CFLAGS) $(INCLUDES) $(LIBPATH32) $(LIBS)

build64:
	sh ../checkenvironment.sh
	${CC} -m64 -o systemSolve systemSolve_new.c -g $(CFLAGS) $(INCLUDES) $(LIBPATH64) $(LIBS)
build64gpu:
	sh ../checkenvironment.sh
	nvcc -arch=sm_20 -m64 -o systemSolve systemSolve_latest.cu -g $(CFLAGS) $(INCLUDES) $(LIBPATH64) $(LIBS)

clean:
	rm -f systemSolve

