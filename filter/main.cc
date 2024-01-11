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

#include "filter.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
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
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_filter(int n, OpenCL &opencl)
{
  int localSize = 256;
  auto input = random_std_vector<float>(n);
  std::vector<float> result(n), expected_result;
  std::vector<int> offsets(n / localSize);
  cl::Kernel map(opencl.program, "map");
  cl::Kernel scanPartial(opencl.program, "scan_partial");
  cl::Kernel scanTotal(opencl.program, "scan_total");
  cl::Kernel scatter(opencl.program, "scatter");

  auto t0 = clock_type::now();
  filter(input, expected_result, [](float x)
           { return x > 0; }); // filter positive numbers

  auto t1 = clock_type::now();
  cl::Buffer inputBuffer(opencl.queue, std::begin(input), std::end(input), false);
  cl::Buffer mapBuffer(opencl.context, CL_MEM_READ_WRITE, sizeof(int) * offsets.size());
  cl::Buffer resultBuffer(opencl.context, CL_MEM_READ_WRITE, sizeof(float) * input.size());
  opencl.queue.flush();

  auto t2 = clock_type::now();

  map.setArg(0, inputBuffer);
  map.setArg(1, mapBuffer);
  opencl.queue.enqueueNDRangeKernel(map, cl::NullRange, cl::NDRange(n), cl::NDRange(localSize));

  scanPartial.setArg(0, mapBuffer);
  opencl.queue.enqueueNDRangeKernel(scanPartial, cl::NullRange, cl::NDRange(n / localSize), cl::NDRange(localSize));

  scanTotal.setArg(0, mapBuffer);
  opencl.queue.enqueueNDRangeKernel(scanTotal, cl::NullRange, cl::NDRange(n / localSize), cl::NDRange(localSize));

  scatter.setArg(0, inputBuffer);
  scatter.setArg(1, mapBuffer);
  scatter.setArg(2, resultBuffer);

  opencl.queue.enqueueNDRangeKernel(scatter, cl::NullRange, cl::NDRange(n), cl::NDRange(localSize));
  opencl.queue.flush();
  auto t3 = clock_type::now();

  cl::copy(opencl.queue, mapBuffer, std::begin(offsets), std::end(offsets));
  cl::copy(opencl.queue, resultBuffer, std::begin(result), std::begin(result) + offsets.back());
  result.resize(offsets.back());
  auto t4 = clock_type::now();

  verify_vector(expected_result, result);
  print("filter", {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3});
}

void opencl_main(OpenCL &opencl)
{
  using namespace std::chrono;
  print_column_names();
  profile_filter(1024 * 1024, opencl);
}

const std::string src = R"(
kernel void map(global float *in, global int *out) {
  const int localSize = get_local_size(0);
  int groupId = get_group_id(0);
  int localId = get_local_id(0);
  local float buff[1024];

  buff[localId] = in[groupId * localSize + localId];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    int cnt = 0;
    for (int j = 0; j < localSize; j++) {
      if (buff[j] > 0)
        cnt++;
    }
    out[groupId] = cnt;
  }
}
kernel void scan_partial(global int *out) {
  const int globalId = get_global_id(0);
  const int localId = get_local_id(0);
  const int localSize = get_local_size(0);

  local int buff[1024];
  buff[localId] = out[globalId];
  barrier(CLK_LOCAL_MEM_FENCE);

  int sum = buff[localId];
  for (int offset = 1; offset < localSize; offset *= 2) {
    if (localId >= offset) {
      sum += buff[localId - offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    buff[localId] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out[globalId] = buff[localId];
}

kernel void scan_total(global int *in) {
  const int groupId = get_group_id(0);
  const int globalSize = get_global_size(0);
  const int localId = get_local_id(0);
  const int localSize = get_local_size(0);
  if (groupId == 0) {
    for (int j = 1; j < globalSize / localSize; j++) {
      in[j * localSize + localId] += in[j * localSize - 1];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
  }
}

kernel void scatter(global float *in, global int *offsets,
                    global float *out) {
  const int localSize = get_local_size(0);
  int groupId = get_group_id(0);
  int localId = get_local_id(0);
  local float buff[1024];

  buff[localId] = in[groupId * localSize + localId];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (localId == 0) {
    int currentIndex = 0;
    if (groupId > 0)
      currentIndex = offsets[groupId - 1];
    for (int j = 0; j < localSize; j++) {
      if (buff[j] > 0) {
        out[currentIndex] = buff[j];
        currentIndex++;
      }
    }
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
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
