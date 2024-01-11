build/filter: main.cc linear-algebra.hh reduce-scan.hh filter.hh
	@mkdir -p build
	g++ -O3 -march=native -fopenmp main.cc -L /opt/rocm/opencl/lib -lOpenCL -o build/filter

run:
	./build/filter > output.txt