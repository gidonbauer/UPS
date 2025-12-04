HEADERS = src/BoundaryConditions.hpp  \
          src/Burgers.hpp             \
          src/Grid.hpp                \
          src/Quadrature.hpp          \
          src/QuadratureTables.hpp    \
          src/TimeIntegrator.hpp      \
          src/Vector.hpp

TARGETS = main burgers scaling

CXX_FLAGS = -Wall -Wextra -pedantic -Wconversion -Wshadow -std=c++23

ifeq (${DEBUG}, 1)
  CXX_FLAGS += -g -O0
else ifeq (${SANITIZE}, 1)
  CXX_FLAGS += -g -O0 -fsanitize=undefined,thread
  # CXX_FLAGS += -g -O0 -fsanitize=address,undefined
else
  CXX_FLAGS += -march=native -O3
  ifeq (${FAST}, 1)
    CXX_FLAGS += -ffast-math -DIGOR_NDEBUG -DNDEBUG
  endif
endif

ifdef IGOR_DIR
  IGOR_INC = -I${IGOR_DIR}
else
  ${error "Need to define the path to Igor library in `IGOR_DIR`."}
endif

all: ${TARGETS}

%: %.cpp ${HEADERS} output
	${CXX} ${CXX_FLAGS} -Isrc/ ${IGOR_INC} -o $@ $<

scaling: scaling.cpp ${HEADERS} output
	${CXX} ${CXX_FLAGS} -fopenmp -Isrc/ ${IGOR_INC} -o $@ $<

output:
	mkdir -p output

clean:
	rm -fr ${TARGETS} ${addsuffix .dSYM, ${TARGETS}}

.PHONY: clean
