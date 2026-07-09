# C++-native PyCATSHOO benchmark (see bench.cpp).
#
# Point PYCATSHOO_DIR at a PyCATSHOO 1.4.x install whose headers match
# its binaries (see the ABI note in README.md). Set it in the env or
# pass it per-invocation: `make PYCATSHOO_DIR=/path/to/pycatshoo-1.4.x`.

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wno-unused-variable

# Fail early (and clearly) if PYCATSHOO_DIR does not point at a 1.4.x
# tree with the PyC headers — avoids the header/binary drift segfault.
pyc_bench: bench.cpp
	@test -n "$(PYCATSHOO_DIR)" || { echo "PYCATSHOO_DIR is not set — see README.md"; exit 1; }
	@test -d "$(PYCATSHOO_DIR)/Core/include/PyC" || { echo "No PyC headers under $(PYCATSHOO_DIR)/Core/include/PyC — need a PyCATSHOO 1.4.x install (see README.md)"; exit 1; }
	$(CXX) $(CXXFLAGS) $< \
	  -I$(PYCATSHOO_DIR)/Core/include/PyC \
	  $(PYCATSHOO_DIR)/Core/lib/libPycatshoo.a \
	  -lpthread -ldl \
	  -o $@

clean:
	rm -f pyc_bench

.PHONY: clean
