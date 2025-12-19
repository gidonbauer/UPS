HEADERS = src/Common/Quadrature.hpp \
          src/Common/QuadratureTables.hpp \
          src/Common/Vector.hpp \
          src/PDE/Advection.hpp \
          src/PDE/BoundaryConditions.hpp \
          src/PDE/Burgers.hpp \
          src/PDE/Grid.hpp \
          src/PDE/Heat.hpp \
          src/PDE/TimeIntegrator.hpp \
          src/ODE/TimeIntegrator.hpp

TARGETS = examples/PDE/burgers \
          examples/PDE/heat \
          examples/PDE/scaling_burgers \
          examples/PDE/scaling_advection \
          examples/ODE/scaling_ode \
          examples/ODE/free_fall \
          examples/ODE/kepler_orbit

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
	${CXX} ${CXX_FLAGS} -fopenmp -Isrc/ ${IGOR_INC} -o $@ $<

output:
	mkdir -p output

clean:
	rm -fr ${TARGETS} ${addsuffix .dSYM, ${TARGETS}}

allclean: clean
	rm -fr output

.PHONY: clean
