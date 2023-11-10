#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
  using namespace std::chrono;
  const auto dt = duration_cast<microseconds>(t1 - t0).count();
  if (dt == 0) {
    return 0;
  }
  return ((n + n + n) * sizeof(float) * 1e-9) / (dt * 1e-6);
}

void print(const char *name, std::array<duration, 5> dt,
           std::array<double, 2> bw) {
  using namespace std::chrono;
  std::cout << std::setw(19) << name;
  for (size_t i = 0; i < 5; ++i) {
    std::stringstream tmp;
    tmp << duration_cast<microseconds>(dt[i]).count() << "us";
    std::cout << std::setw(20) << tmp.str();
  }
  for (size_t i = 0; i < 2; ++i) {
    std::stringstream tmp;
    tmp << bw[i] << "GB/s";
    std::cout << std::setw(20) << tmp.str();
  }
  std::cout << '\n';
}

void print_column_names() {
  std::cout << std::setw(19) << "function";
  std::cout << std::setw(20) << "OpenMP";
  std::cout << std::setw(20) << "OpenCL total";
  std::cout << std::setw(20) << "OpenCL copy-in";
  std::cout << std::setw(20) << "OpenCL kernel";
  std::cout << std::setw(20) << "OpenCL copy-out";
  std::cout << std::setw(20) << "OpenMP bandwidth";
  std::cout << std::setw(20) << "OpenCL bandwidth";
  std::cout << '\n';
}

struct OpenCL {
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
};

void profile_reduce(int n, OpenCL &opencl) {
  auto a = random_vector<float>(n);
  Vector<float> result(1), expected_result(1);
  int global_size = n;
  int local_size = 128;
  std::vector<cl::Buffer> tiles;
  opencl.queue.flush();
  cl::Kernel kernel_reduce(opencl.program, "reduce");
  cl::Kernel kernel_complete(opencl.program, "reduce_complete");
  auto t0 = clock_type::now();
  expected_result[0] = reduce(a);
  auto t1 = clock_type::now();
  cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
  tiles.emplace_back(d_a);
  cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, 1 * sizeof(float));
  auto t2 = clock_type::now();
  int i = 0;
  while (global_size > local_size) {
    auto next_global_size = global_size / local_size;
    cl::Buffer tile_result(opencl.context, CL_MEM_READ_WRITE,
                           next_global_size * sizeof(float));
    tiles.emplace_back(tile_result);
    kernel_reduce.setArg(0, tiles[i]);
    kernel_reduce.setArg(1, tiles[i + 1]);
    kernel_reduce.setArg(2, global_size);
    kernel_reduce.setArg(3, cl::Local(local_size * sizeof(float)));
    opencl.queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange,
                                      cl::NDRange(global_size),
                                      cl::NDRange(local_size));
    opencl.queue.flush();
    global_size = global_size / local_size;
    i++;
  }
  kernel_complete.setArg(0, tiles[i]);
  kernel_complete.setArg(1, global_size);
  kernel_complete.setArg(2, d_result);
  opencl.queue.enqueueNDRangeKernel(kernel_complete, cl::NullRange,
                                    cl::NDRange(global_size), cl::NullRange);
  opencl.queue.flush();
  auto t3 = clock_type::now();
  cl::copy(opencl.queue, d_result, begin(result), end(result));
  auto t4 = clock_type::now();
  verify_vector(expected_result, result, (float)(1e2));
  print("reduce", {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
        {bandwidth(n * n + n + n, t0, t1), bandwidth(n * n + n + n, t2, t3)});
}

void profile_scan_inclusive(int n, OpenCL &opencl) {
  auto a = random_vector<float>(n);
  Vector<float> result(a), expected_result(a);
  int global_size = n;
  int local_size = 64;
  std::vector<cl::Buffer> tiles;
  std::vector<int> sizes;
  cl::Kernel kernel_scan(opencl.program, "scan_inclusive");
  cl::Kernel kernel_complete(opencl.program, "scan_complete");
  auto t0 = clock_type::now();
  scan_inclusive(expected_result);
  auto t1 = clock_type::now();
  cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
  tiles.emplace_back(d_a);
  auto t2 = clock_type::now();
  int i = 0;
  while (global_size > 1) {
    int next_global_size = (global_size + local_size - 1) / local_size;

    cl::Buffer tile_result(opencl.context, CL_MEM_READ_WRITE,
                           (global_size + local_size) * sizeof(float));
    tiles.emplace_back(tile_result);
    sizes.emplace_back(global_size);

    kernel_scan.setArg(0, tiles[i]);
    kernel_scan.setArg(1, tiles[i + 1]);
    kernel_scan.setArg(2, cl::Local(local_size * sizeof(float)));
    kernel_scan.setArg(3, global_size);
    opencl.queue.flush();

    opencl.queue.enqueueNDRangeKernel(
        kernel_scan, cl::NullRange, cl::NDRange(next_global_size * local_size),
        cl::NDRange(local_size));
    opencl.queue.flush();

    i++;
    global_size = next_global_size;
  }
  for (int j = i - 1; j >= 1; j--) {
    kernel_complete.setArg(0, tiles[j - 1]);
    kernel_complete.setArg(1, tiles[j]);
    kernel_complete.setArg(2, sizes[j - 1]);
    opencl.queue.flush();

    opencl.queue.enqueueNDRangeKernel(
        kernel_complete, cl::NullRange,
        cl::NDRange(((sizes[j - 1] + local_size - 1) / local_size) *
                    local_size),
        cl::NDRange(local_size));
    opencl.queue.flush();
  }
  auto t3 = clock_type::now();
  opencl.queue.enqueueReadBuffer(tiles[0], true, 0,
                                 result.size() * sizeof(float), begin(result));
  opencl.queue.flush();
  auto t4 = clock_type::now();
  verify_vector(expected_result, result, (float)(1e4));
  print("scan-inclusive", {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
        {bandwidth(n * n + n * n + n * n, t0, t1),
         bandwidth(n * n + n * n + n * n, t2, t3)});
}

void opencl_main(OpenCL &opencl) {
  using namespace std::chrono;
  print_column_names();
  profile_reduce(1024 * 1024 * 10, opencl);
  profile_scan_inclusive(1024 * 1024 * 10, opencl);
}

const std::string src = R"(
kernel void reduce(global float* a,
                    global float* result,
                   int n,         
                   local float* tileResult) {
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    if (global_id >= n) {
        tileResult[local_id] = .0;
    } else {
        tileResult[local_id] = a[global_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = local_size/2; offset>0; offset /=2) {
        if (local_id < offset)
            tileResult[local_id] += tileResult[local_id + offset];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
       	result[group_id] = tileResult[0];
    }
}


kernel void reduce_complete(global float* a, 
                            int n,
                            global float* result) {
    int global_id = get_global_id(0);
    if (global_id == 0) {
         for (int i = 0; i < n; i++) {
            result[0] += a[i];
          }
    }
}

kernel void scan_inclusive(global float* a,
                            global float* result,
                            local float* tileResult,
                            int n) {
    int local_id = get_local_id(0); // номер потока в группе
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    if (global_id >= n) {
        tileResult[local_id] = .0;
    } else {
        tileResult[local_id] = a[global_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int offset = 1; offset < local_size; offset *= 2) {
        float sum = 0;
        if(local_id >= offset && global_id < n) {
            sum += tileResult[local_id - offset];
            tileResult[local_id] += sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n){
        a[global_id] = tileResult[local_id];
    }

    if (local_id == 0) {        
       	result[group_id] = tileResult[local_size-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void scan_complete(global float* a, 
                            global float* tileResult,
                            int n) {
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    if (global_id >= local_size && global_id < n){
        a[global_id] += tileResult[group_id-1];
    }
}
)";

int main() {
  try {
    // find OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
      std::cerr << "Unable to find OpenCL platforms\n";
      return 1;
    }
    cl::Platform platform = platforms[0];
    std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>()
              << '\n';
    // create context
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                          (cl_context_properties)platform(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);
    // get all devices associated with the context
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device device = devices[0];
    std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
    cl::Program program(context, src);
    // compile the programme
    try {
      program.build(devices);
    } catch (const cl::Error &err) {
      for (const auto &device : devices) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << log;
      }
      throw;
    }
    cl::CommandQueue queue(context, device);
    OpenCL opencl{platform, device, context, program, queue};
    opencl_main(opencl);
  } catch (const cl::Error &err) {
    std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
    std::cerr << "Search cl.h file for error code (" << err.err()
              << ") to understand what it means:\n";
    std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/"
                 "CL/cl.h\n";
    return 1;
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    return 1;
  }
  return 0;
}