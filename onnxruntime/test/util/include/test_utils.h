// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/framework_common.h"
#include "core/framework/execution_provider.h"

#include <memory>
#include <string>
#include <vector>

namespace onnxruntime {
class Graph;

namespace test {

// struct to hold some verification params for RunAndVerifyOutputsWithEP
struct EPVerificationParams {
  // Verify the entire graph is taken by the EP
  // if this is set to false, then will verify that at least one node is assigned to 'execution_provider'
  bool verify_entire_graph_use_ep{false};

  // Some EP may use different rounding than ORT CPU EP, which may cause a bigger abs error than
  // the default of 1e-5f, especially for scenarios such as [Q -> Quantized op -> DQ]
  // Set this only if this is necessary
  float fp32_abs_err = 1e-5f;
};

// return number of nodes in the Graph and any subgraphs that are assigned to the specified execution provider
int CountAssignedNodes(const Graph& current_graph, const std::string& ep_type);

// run the model using the CPU EP to get expected output, comparing to the output when the 'execution_provider'
// is enabled. requires that at least one node is assigned to 'execution_provider'
void RunAndVerifyOutputsWithEP(const ORTCHAR_T* model_path,
                               const char* log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds,
                               const EPVerificationParams& params = EPVerificationParams());

// helper function that takes in model_data
void RunAndVerifyOutputsWithEP(const std::string& model_data,
                               const char* log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds,
                               const EPVerificationParams& params = EPVerificationParams());
}  // namespace test
}  // namespace onnxruntime
