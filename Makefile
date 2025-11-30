HEADERS = src/Quadrature.hpp        \
          src/QuadratureTables.hpp  \
          src/Vector.hpp


CXX_FLAGS = -Wall -Wextra -pedantic -Wconversion -Wshadow -std=c++23

ifeq (${DEBUG}, 1)
	CXX_FLAGS += -g -O0
else
	CXX_FLAGS += -march=native -O3
endif

IGOR_INC = -I${IGOR_DIR}

main: main.cpp ${HEADERS} output
	${CXX} ${CXX_FLAGS} -Isrc/ ${IGOR_INC} -o $@ $<

output:
	mkdir -p output

clean:
	rm -fr main main.dSYM

.PHONY: clean
