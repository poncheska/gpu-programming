build/ray-tracing: main.cc theora.cc vector.hh ray.hh color.hh scene.hh theora.hh
	@mkdir -p build
	g++ -O3 -march=native -fopenmp $(shell pkg-config --cflags theoraenc) main.cc theora.cc -L /opt/rocm/opencl/lib -lOpenCL $(shell pkg-config --libs theoraenc) -o build/ray-tracing

run:
	./build/ray-tracing > output.txt