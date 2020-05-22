#include <iostream>
#include <chrono>
#include <torch/script.h>
#include <torch/torch.h>

int main() {
  torch::jit::script::Module model = torch::jit::load("./pth/traced.pt");

  int N = 1;
  int L = 8;
  int C = 6;

  at::Tensor input = torch::randn({N, L, C});

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input);
  inputs.push_back(torch::randn({2, N, 128}));
  inputs.push_back(torch::randn({2, N, 128}));

  auto t1 = std::chrono::system_clock::now();
  for (int i = 0; i < 1000; ++i) {
    auto output = model.forward(inputs).toTensor();
  }
  auto t2 = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = t2 - t1;
  std::cout << diff.count() * 1000 << std::endl;
}
