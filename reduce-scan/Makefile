build/reduce-scan: main.cc linear-algebra.hh reduce-scan.hh
	@mkdir -p build
	g++ -O3 -march=native -fopenmp main.cc -L /opt/rocm/opencl/lib -lOpenCL -o build/reduce-scan

run:
	./build/reduce-scan > output.txt