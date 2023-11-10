build/super-boring-task: main.cc linear-algebra.hh
	@mkdir -p build
	g++ -O3 -march=native -fopenmp main.cc -L /opt/rocm/opencl/lib -lOpenCL -o build/super-boring-task
	
run:
	./build/super-boring-task > output.txt
