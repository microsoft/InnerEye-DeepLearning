## Pytest and testing on CPU and GPU machines

All non-trivial proposed changes to the code base should be accompanied by tests.

Each PullRequest build will run all tests in the repository on CPU machines. One full test run is executed on a Windows
agent, one on a Linux agent.

In addition, `pytest` will be run as part of the small AzureML job (smoke test) that is part of the PR build. 
In that test run, only specific tests will be executed. At present, it will be the tests that are marked 
with the `pytest` mark `gpu`, `cpu_and_gpu` or `azureml`. The AzureML job executes on a GPU VM, hence you can have 
tests for GPU-specific capabilities.

To mark one of your tests as requiring a GPU, prefix the test as follows:

    @pytest.mark.gpu
    def test_my_gpu_code() -> None:
       ...
       
Similarly, use the mark `cpu_and_gpu` for tests that should be run twice, once on the CPU and once on the GPU. For 
tests that do not require a GPU, but have longer running times and are better suited to run on the PR build, use the
mark `azureml`.
