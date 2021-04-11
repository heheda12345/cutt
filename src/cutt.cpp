/******************************************************************************
MIT License

Copyright (c) 2016 Antti-Pekka Hynninen
Copyright (c) 2016 Oak Ridge National Laboratory (UT-Batelle)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/
#include <cuda.h>
#include <list>
#include <unordered_map>
#include "CudaUtils.h"
#include "cuttplan.h"
#include "cuttkernel.h"
#include "cuttTimer.h"
#include "cutt.h"
// #include <chrono>

// Hash table to store the plans
static std::unordered_map< cuttHandle, cuttPlan_t* > planStorage;

// Current handle
static cuttHandle curHandle = 0;

// Table of devices that have been initialized
static std::unordered_map<int, cudaDeviceProp> deviceProps;

#ifdef _OPENMP
#include <omp.h>
omp_lock_t lockPlanStorage;
omp_lock_t lockCurHandle;
omp_lock_t lockDeviceProps;
#endif

cuttResult cuttInit() {
#ifdef _OPENMP
  omp_init_lock(&lockPlanStorage);
  omp_init_lock(&lockCurHandle);
  omp_init_lock(&lockDeviceProps);
#endif
  return CUTT_SUCCESS;
}

// Checks prepares device if it's not ready yet and returns device properties
// Also sets shared memory configuration
void getDeviceProp(int& deviceID, cudaDeviceProp &prop) {
#ifdef _OPENMP
  omp_set_lock(&lockCurHandle);
  cudaCheck(cudaGetDevice(&deviceID));
  auto it = deviceProps.find(deviceID);
  if (it == deviceProps.end()) {
    // Get device properties and store it for later use
    cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
    // cuttKernelSetSharedMemConfig();
    deviceProps.insert({deviceID, prop});
  } else {
    prop = it->second;
  }
  omp_unset_lock(&lockCurHandle);
#else
  cudaCheck(cudaGetDevice(&deviceID));
  auto it = deviceProps.find(deviceID);
  if (it == deviceProps.end()) {
    // Get device properties and store it for later use
    cudaCheck(cudaGetDeviceProperties(&prop, deviceID));
    // cuttKernelSetSharedMemConfig();
    deviceProps.insert({deviceID, prop});
  } else {
    prop = it->second;
  }
#endif
}

cuttResult cuttPlanCheckInput(int rank, int* dim, int* permutation, size_t sizeofType) {
  // Check sizeofType
  if (sizeofType != 4 && sizeofType != 8 && sizeofType != 16) return CUTT_INVALID_PARAMETER;
  // Check rank
  if (rank <= 1) return CUTT_INVALID_PARAMETER;
  // Check dim[]
  for (int i=0;i < rank;i++) {
    if (dim[i] <= 1) return CUTT_INVALID_PARAMETER;
  }
  // Check permutation
  bool permutation_fail = false;
  int* check = new int[rank];
  for (int i=0;i < rank;i++) check[i] = 0;
  for (int i=0;i < rank;i++) {
    if (permutation[i] < 0 || permutation[i] >= rank || check[permutation[i]]++) {
      permutation_fail = true;
      break;
    }
  }
  delete [] check;
  if (permutation_fail) return CUTT_INVALID_PARAMETER;  

  return CUTT_SUCCESS;
}

cuttResult cuttPlan(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream, bool activate) {

#ifdef ENABLE_NVTOOLS
  gpuRangeStart("init");
#endif

  // Check that input parameters are valid
  cuttResult inpCheck = cuttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

#ifdef _OPENMP
  omp_set_lock(&lockCurHandle);
  // Create new handle
  *handle = curHandle;
  curHandle++;
  // Check that the current handle is available (it better be!)
  omp_unset_lock(&lockCurHandle);
  omp_set_lock(&lockPlanStorage);
  size_t count = planStorage.count(*handle);
  omp_unset_lock(&lockPlanStorage);
  if (count != 0) return CUTT_INTERNAL_ERROR;
#else
  // Create new handle
  *handle = curHandle;
  curHandle++;
  // Check that the current handle is available (it better be!)
  if (planStorage.count(*handle) != 0) return CUTT_INTERNAL_ERROR;
#endif

  // Prepare device
  int deviceID;
  cudaDeviceProp prop;
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<cuttPlan_t> plans;
  // if (rank != redDim.size()) {
  //   if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
  // }

  // // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;

#if 0
  if (!cuttKernelDatabase(deviceID, prop)) return CUTT_INTERNAL_ERROR;
#endif

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("createPlans");
#endif

  // std::chrono::high_resolution_clock::time_point plan_start;
  // plan_start = std::chrono::high_resolution_clock::now();

  if (!cuttPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return CUTT_INTERNAL_ERROR;

  // std::chrono::high_resolution_clock::time_point plan_end;
  // plan_end = std::chrono::high_resolution_clock::now();
  // double plan_duration = std::chrono::duration_cast< std::chrono::duration<double> >(plan_end - plan_start).count();
  // printf("createPlans took %lf ms\n", plan_duration*1000.0);

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("countCycles");
#endif

  // Count cycles
  for (auto it=plans.begin();it != plans.end();it++) {
    if (!it->countCycles(prop, 10)) return CUTT_INTERNAL_ERROR;
  }

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
  gpuRangeStart("rest");
#endif

  // Choose the plan
  std::list<cuttPlan_t>::iterator bestPlan = choosePlanHeuristic(plans);
  if (bestPlan == plans.end()) return CUTT_INTERNAL_ERROR;

  // bestPlan->print();

  // Create copy of the plan outside the list
  cuttPlan_t* plan = new cuttPlan_t();
  // NOTE: No deep copy needed here since device memory hasn't been allocated yet
  *plan = *bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan->nullDevicePointers();

  if (activate) {
    // Set stream
    plan->setStream(stream);

    // Activate plan
    plan->activate();
  } else {
    plan->nullDevicePointers();
  }

#ifdef _OPENMP
  omp_set_lock(&lockPlanStorage);
  // Insert plan into storage
  planStorage.insert( {*handle, plan} );
  omp_unset_lock(&lockPlanStorage);
#else
  // Insert plan into storage
  planStorage.insert( {*handle, plan} );
#endif

#ifdef ENABLE_NVTOOLS
  gpuRangeStop();
#endif

  return CUTT_SUCCESS;
}

cuttResult cuttActivatePlan(cuttHandle* newHandle, cuttHandle oldHandle, cudaStream_t stream, int deviceID) {
#ifdef _OPENMP
  omp_set_lock(&lockPlanStorage);
  cuttPlan_t* oldPlan = planStorage[oldHandle];
  omp_unset_lock(&lockPlanStorage);
#else
  cuttPlan_t* oldPlan = planStorage[oldHandle];
#endif
  cuttPlan_t* newPlan = new cuttPlan_t();
  *newPlan = *oldPlan;

#ifdef _OPENMP
  omp_set_lock(&lockCurHandle);
  *newHandle = curHandle;
  curHandle++;
  omp_unset_lock(&lockCurHandle);
#else
  *newHandle = curHandle;
  curHandle++;
#endif

  newPlan->setStream(stream);
  newPlan->activate();
  newPlan->deviceID = deviceID;

#ifdef _OPENMP
  omp_set_lock(&lockPlanStorage);
  planStorage.insert( {*newHandle, newPlan} );
  omp_unset_lock(&lockPlanStorage);
#else
  planStorage.insert( {*newHandle, newPlan} );
#endif

  return CUTT_SUCCESS;
}

cuttResult cuttPlanMeasure(cuttHandle* handle, int rank, int* dim, int* permutation, size_t sizeofType,
  cudaStream_t stream, void* idata, void* odata) {

  // Check that input parameters are valid
  cuttResult inpCheck = cuttPlanCheckInput(rank, dim, permutation, sizeofType);
  if (inpCheck != CUTT_SUCCESS) return inpCheck;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

#ifdef _OPENMP
  omp_set_lock(&lockCurHandle);
  // Create new handle
  *handle = curHandle;
  curHandle++;
  omp_unset_lock(&lockCurHandle);
  // Check that the current handle is available (it better be!)
  omp_set_lock(&lockPlanStorage);
  size_t count = planStorage.count(*handle);
  omp_unset_lock(&lockPlanStorage);
  if (count != 0) return CUTT_INTERNAL_ERROR;
#else
  // Create new handle
  *handle = curHandle;
  curHandle++;
  // Check that the current handle is available (it better be!)
  if (planStorage.count(*handle) != 0) return CUTT_INTERNAL_ERROR;
#endif

  // Prepare device
  int deviceID;
  cudaDeviceProp prop;
  getDeviceProp(deviceID, prop);

  // Reduce ranks
  std::vector<int> redDim;
  std::vector<int> redPermutation;
  reduceRanks(rank, dim, permutation, redDim, redPermutation);

  // Create plans from reduced ranks
  std::list<cuttPlan_t> plans;
#if 0
  // if (rank != redDim.size()) {
    if (!createPlans(redDim.size(), redDim.data(), redPermutation.data(), sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
  // }

  // Create plans from non-reduced ranks
  // if (!createPlans(rank, dim, permutation, sizeofType, prop, plans)) return CUTT_INTERNAL_ERROR;
#else
  if (!cuttPlan_t::createPlans(rank, dim, permutation, redDim.size(), redDim.data(), redPermutation.data(), 
    sizeofType, deviceID, prop, plans)) return CUTT_INTERNAL_ERROR;
#endif

  // // Count cycles
  // for (auto it=plans.begin();it != plans.end();it++) {
  //   if (!it->countCycles(prop, 10)) return CUTT_INTERNAL_ERROR;
  // }

  // // Count the number of elements
  size_t numBytes = sizeofType;
  for (int i=0;i < rank;i++) numBytes *= dim[i];

  // Choose the plan
  double bestTime = 1.0e40;
  auto bestPlan = plans.end();
  Timer timer;
  std::vector<double> times;
  for (auto it=plans.begin();it != plans.end();it++) {
    // Activate plan
    it->activate();
    // Clear output data to invalidate caches
    set_device_array<char>((char *)odata, -1, numBytes);
    cudaCheck(cudaDeviceSynchronize());
    timer.start();
    // Execute plan
    if (!cuttKernel(*it, idata, odata)) return CUTT_INTERNAL_ERROR;
    timer.stop();
    double curTime = timer.seconds();
    // it->print();
    // printf("curTime %1.2lf\n", curTime*1000.0);
    times.push_back(curTime);
    if (curTime < bestTime) {
      bestTime = curTime;
      bestPlan = it;
    }
  }
  if (bestPlan == plans.end()) return CUTT_INTERNAL_ERROR;

  // bestPlan = plans.begin();

  // printMatlab(prop, plans, times);
  // findMispredictionBest(plans, times, bestPlan, bestTime);
  // bestPlan->print();

  // Create copy of the plan outside the list
  cuttPlan_t* plan = new cuttPlan_t();
  *plan = *bestPlan;
  // Set device pointers to NULL in the old copy of the plan so
  // that they won't be deallocated later when the object is destroyed
  bestPlan->nullDevicePointers();

  // Set stream
  plan->setStream(stream);

  // Activate plan
  plan->activate();

  // Insert plan into storage
#ifdef _OPENMP
  omp_set_lock(&lockPlanStorage);
  planStorage.insert( {*handle, plan} );
  omp_unset_lock(&lockPlanStorage);
#else
  planStorage.insert( {*handle, plan} );
#endif

  return CUTT_SUCCESS;
}

cuttResult cuttDestroy(cuttHandle handle) {
#ifdef _OPENMP
  omp_set_lock(&lockPlanStorage);
  auto it = planStorage.find(handle);
  omp_unset_lock(&lockPlanStorage);
#else
  auto it = planStorage.find(handle);
#endif
  if (it == planStorage.end()) return CUTT_INVALID_PLAN;
  // Delete instance of cuttPlan_t
  delete it->second;
  // Delete entry from plan storage
#ifdef _OPENMP
  omp_set_lock(&lockPlanStorage);
  planStorage.erase(it);
  omp_unset_lock(&lockPlanStorage);
#else
  planStorage.erase(it);
#endif
  return CUTT_SUCCESS;
}

cuttResult cuttExecute(cuttHandle handle, void* idata, void* odata) {
#ifdef _OPENMP
  omp_set_lock(&lockPlanStorage);
  auto it = planStorage.find(handle);
  omp_unset_lock(&lockPlanStorage);
#else
  auto it = planStorage.find(handle);
#endif
  if (it == planStorage.end()) return CUTT_INVALID_PLAN;

  if (idata == odata) return CUTT_INVALID_PARAMETER;

  cuttPlan_t& plan = *(it->second);

  int deviceID;
  cudaCheck(cudaGetDevice(&deviceID));
  if (deviceID != plan.deviceID) return CUTT_INVALID_DEVICE;

  if (!cuttKernel(plan, idata, odata)) return CUTT_INTERNAL_ERROR;
  return CUTT_SUCCESS;
}
