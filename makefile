PETSC_DIR=/users/tany3/petsc-2.3.3-p16

PETSC_ARCH=linux-gnu-cxx-debug

MPI_PATH=/usr/local/mpich3/3.1.2-thread-multiple/bin
#MPI_PATH=/usr/local/mpich2/latest/bin

#CXX=mpicxx
CXX=/usr/local/mpich3/3.1.2-thread-multiple/mpicxx
#CXX=/usr/local/mpich2/latest/bin/mpicxx

#If there are any .hpp files
HS = fem.hpp share.hpp
STUFF = -DMPICH_IGNORE_CXX_SEEK -I. -I/usr/include/ -I/usr/local/include\
	-I${PETSC_DIR}/include\
	-I${PETSC_DIR}/bmake/${PETSC_ARCH}\
	-I${MPI_PATH}/include \
	-I/usr/include/X11
OBJECTS = mycalls.o assem.o bookkeep.o

LDFLAGS =\
	-L$(PETSC_DIR)/lib/$(PETSC_ARCH)\
	-L${MPI_PATH}/lib\
	-L${MPI_PATH}/lib64\
	-lpetscdm\
	-lpetscksp\
	-lpetscmat\
	-lpetscvec\
	-llapack\
	-lpthread\
	-lpetsc

LDLIBS = -lglib -lm 

LDLIBS += -framework IOK

include ${PETSC_DIR}/bmake/common/base

all: ${OBJECTS}

mycalls.o: share.hpp mycalls.hpp mycalls.cpp
	${CXX} ${STUFF} -g -c mycalls.cpp

assem.o: fem.hpp assem.cpp
	${CXX} ${STUFF} -g -c assem.cpp

bookkeep.o: fem.hpp bookkeep.cpp
	${CXX} ${STUFF} -g -c bookkeep.cpp

main.o: main.cpp ${HS}
	${CXX} ${STUFF} -g -c main.cpp

radiation: main.o mycalls.o
	${CXX} -Dradiation ${STUFF} -DMPI_VERSION -o driver main.o mycalls.o ${LDFLAGS}

driver: main.o mycalls.o
	${CXX} ${STUFF} -DMPI_VERSION -o driver main.o mycalls.o ${LDFLAGS}

clean: 
	\rm *.o results.m

include ${PETSC_DIR}/bmake/common/test