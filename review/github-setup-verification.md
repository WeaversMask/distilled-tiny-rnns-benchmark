# Historical launcher verification

The historical pilot launcher completed all five code cells on CPU with **106 passing tests** and one CUDA-only skip. A separate internal check passed **25 focused tests** covering archive validation, safe redirects and exact source pinning. Both suites used a deliberately malformed duplicate-member ZIP fixture.

The runtime source digest was `e161fb5f9f7215c05d7fa1bfe22edf3225809c5c8258013e75fa2be051b95436`. The download matched that digest. [Current Colab instructions](../README.md#run-in-colab)
