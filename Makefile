HEADERS = src/Quadrature.hpp        \
          src/QuadratureTables.hpp  \
          src/Vector.hpp
TARGETS = main burgers

CXX_FLAGS = -Wall -Wextra -pedantic -Wconversion -Wshadow -std=c++23

ifeq (${DEBUG}, 1)
	CXX_FLAGS += -g -O0
else
	CXX_FLAGS += -march=native -O3
endif

IGOR_INC = -I${IGOR_DIR}

all: ${TARGETS}

%: %.cpp ${HEADERS} output
	${CXX} ${CXX_FLAGS} -Isrc/ ${IGOR_INC} -o $@ $<

output:
	mkdir -p output

clean:
	rm -fr ${TARGETS} ${addsuffix .dSYM, ${TARGETS}}

.PHONY: clean
