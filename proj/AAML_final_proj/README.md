# AAML Final project 

## Team Member
+ 周哲瑋 313551118
+ 陳冠霖 313553054


## Methodology

Our approach builds upon the implementation from **Lab 5** by utilizing a 4×4 systolic array to accelerate convolution operations. In addition, we addressed the time-consuming initialization issue inherent in the original design, resulting in a significant performance improvement—reducing the golden test runtime to **53M cycles**.

### Optimizations

1. **Matrix Tiling Enhancement**  
   We doubled the size of matrix tiling, further reducing the golden test runtime to the current **50M cycles**.

2. **Build Optimization**  
   By using the command `make prog EXTRA_LITEX_ARGS="--cpu-variant=perf+cfu"`, we observed an additional **61M cycle** improvement compared to the default `make prog` command.

### Execution Instructions

To replicate these results, follow the steps below:

1. Build the program with the optimized CPU variant:
   ```bash
   make prog EXTRA_LITEX_ARGS="--cpu-variant=perf+cfu"
   ```
2. Load the program:
   ```bash
   make load
   ```
3. Run the evaluation script to measure latency:
   ```bash
   python eval_script.py
   ```
   or  
   ```bash
   python3 eval_script.py
   ```

These enhancements collectively demonstrate significant improvements in system performance and provide a robust method for accelerating convolution operations.


## Final Project Source
https://nycu-caslab.github.io/AAML2024/project/final_project.html


