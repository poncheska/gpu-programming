build/sph: main.cc theora.cc opengl.hh theora.hh vector.hh
	@mkdir -p build
	g++ -O3 -march=native -fopenmp $(shell pkg-config --cflags theoraenc glut glu gl glew) main.cc theora.cc -L /opt/rocm/opencl/lib -lOpenCL $(shell pkg-config --libs theoraenc glut glu gl glew) -o build/sph

run: build/sph
	Xvfb :123 -screen 0 800x600x24 &
	env DISPLAY=:123 ./build/sph

vglrun: build/sph
	env LD_LIBRARY_PATH=$(LIBRARY_PATH) vglrun ./build/sph

clean:
	rm -rf build

.PHONY: run clean
