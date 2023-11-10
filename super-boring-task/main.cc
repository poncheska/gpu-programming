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

void profile_vector_times_vector(int n, OpenCL &opencl) {
  auto a = random_vector<float>(n);
  auto b = random_vector<float>(n);
  Vector<float> result(n), expected_result(n);
  opencl.queue.flush();
  cl::Kernel kernel(opencl.program, "vector_times_vector");
  auto t0 = clock_type::now();
  vector_times_vector(a, b, expected_result);
  auto t1 = clock_type::now();
  cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
  cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
  cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE,
                      result.size() * sizeof(float));
  kernel.setArg(0, d_a);
  kernel.setArg(1, d_b);
  kernel.setArg(2, d_result);
  opencl.queue.flush();
  auto t2 = clock_type::now();
  opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n),
                                    cl::NullRange);
  opencl.queue.flush();
  auto t3 = clock_type::now();
  cl::copy(opencl.queue, d_result, begin(result), end(result));
  auto t4 = clock_type::now();
  verify_vector(expected_result, result);
  print("vector-times-vector", {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
        {bandwidth(n + n + n, t0, t1), bandwidth(n + n + n, t2, t3)});
}

void profile_matrix_times_vector(int n, OpenCL &opencl) {
  auto a = random_matrix<float>(n, n);
  auto b = random_vector<float>(n);
  Vector<float> result(n), expected_result(n);
  opencl.queue.flush();
  cl::Kernel kernel(opencl.program, "matrix_times_vector");
  auto t0 = clock_type::now();
  matrix_times_vector(a, b, expected_result);
  auto t1 = clock_type::now();
  cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
  cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
  cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE,
                      result.size() * sizeof(float));
  kernel.setArg(0, d_a);
  kernel.setArg(1, d_b);
  kernel.setArg(2, d_result);
  opencl.queue.flush();
  auto t2 = clock_type::now();
  opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n),
                                    cl::NullRange);
  opencl.queue.flush();
  auto t3 = clock_type::now();
  cl::copy(opencl.queue, d_result, begin(result), end(result));
  auto t4 = clock_type::now();
  verify_vector(expected_result, result, 1e-1f);
  print("matrix-times-vector", {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
        {bandwidth(n * n + n + n, t0, t1), bandwidth(n * n + n + n, t2, t3)});
}

void profile_matrix_times_matrix(int n, OpenCL &opencl) {
  auto a = random_matrix<float>(n, n);
  auto b = random_matrix<float>(n, n);
  Matrix<float> result(n, n), expected_result(n, n);
  opencl.queue.flush();
  cl::Kernel kernel(opencl.program, "matrix_times_matrix");

  auto t0 = clock_type::now();
  matrix_times_matrix(a, b, expected_result);

  auto t1 = clock_type::now();
  cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
  cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
  cl::Buffer d_result(opencl.context, CL_MEM_WRITE_ONLY,
                      result.size() * sizeof(float));
  int local_size = sqrt(opencl.device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

  while ((n * n) % (local_size * local_size) != 0) {
    local_size--;
  }

  kernel.setArg(0, d_a);
  kernel.setArg(1, d_b);
  kernel.setArg(2, d_result);
  kernel.setArg(3, n);
  kernel.setArg(4, local_size * local_size * sizeof(float), NULL);
  kernel.setArg(5, local_size * local_size * sizeof(float), NULL);
  opencl.queue.finish();

  auto t2 = clock_type::now();
  opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n, n),
                                    cl::NDRange(local_size, local_size));
  opencl.queue.finish();

  auto t3 = clock_type::now();
  cl::copy(opencl.queue, d_result, begin(result), end(result));

  auto t4 = clock_type::now();
  verify_matrix(expected_result, result, 1e-1f);
  print("matrix-times-matrix", {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
        {bandwidth(n * n + n * n + n * n, t0, t1),
         bandwidth(n * n + n * n + n * n, t2, t3)});
}

void opencl_main(OpenCL &opencl) {
  using namespace std::chrono;
  print_column_names();
  profile_vector_times_vector(1024 * 1024 * 10, opencl);
  profile_matrix_times_vector(1024 * 10, opencl);
  profile_matrix_times_matrix(1024, opencl);
}

const std::string src = R"(
kernel void vector_times_vector(global float* a,
                                global float* b,
                                global float* result) {
    const int i = get_global_id(0);
    result[i] = a[i] * b[i];
}

kernel void matrix_times_vector(global const float* a,
                                global const float* b,
                                global float* result) {
    const int i = get_global_id(0);
    const int n = get_global_size(0);
    float sum = 0;
    for (int j=0; j<n; ++j) {
        sum += a[i*n + j]*b[j];
    }
    result[i] = sum;
}

kernel void matrix_times_matrix(global float* a,
                                global float* b,
                                global float* result, 
                                int n,
                                local float* loc_block_A,
                                local float* loc_block_B) {
    int local_size = get_local_size(0);

    int group_id_0 = get_group_id(0);
    int group_id_1 = get_group_id(1);

    int local_id_0 = get_local_id(0);
    int local_id_1 = get_local_id(1);

    int a_start = n * local_size * group_id_1;
    int a_end   = a_start + n - 1;
    int a_step  = local_size;

    int b_start = local_size * group_id_0;
    int b_step  = local_size * n;

    float resultSub = 0.0f;

    for (int i = a_start, j = b_start; i <= a_end; i += a_step, j += b_step) {
        loc_block_A[local_id_0 + local_id_1*local_size] = a[i + n*local_id_1 + local_id_0];
        loc_block_B[local_id_0 + local_id_1*local_size] = b[j + n*local_id_1 + local_id_0];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < local_size; ++k) {
            resultSub += loc_block_A[k + local_id_1*local_size] * loc_block_B[local_id_0 + k*local_size];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[get_global_id(1) * get_global_size(0) + get_global_id(0)] = resultSub;
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
