# README

## Overview
This repository provides **three Python scripts** that implement three bilateral gradual semantics for **weighted argumentation graph**:

1. **ARM_semantics_calculation.py** – AR-max-based semantics  
2. **ARC_semantics_calculation.py** – AR-card-based semantics  
3. **ARH_semantics_calculation.py** – AR-hybrid-based semantics  

Each script reads `.bag` files, computes the acceptability \(\sigma^+(a)\) and rejectability \(\sigma^-(a)\) for each argument, iterates until convergence, and outputs both per-iteration details and final results. They serve as reference implementations to replicate or verify empirical findings related to bilateral gradual semantics in weighted argumentation.


---

## Usage

1. **Setup folders**  
   - Create a directory (e.g., `WAGTest/`) and place the `.py` scripts (ARM, ARC, ARH) there.  
   - Inside `WAGTest/`, create a subfolder named `benchmarks/` for your `.bag` files.

2. **Prepare .bag files**  
   - Store your `.bag` files in `WAGTest/benchmarks/`.  
   - Optionally, you can organize them into subfolders like `WAGTest/benchmarks/100/`, etc.

3. **Run the script**  
   - Open a terminal in `WAGTest/` and execute one of the following:
     ```bash
     python ARM_semantics_calculation.py
     python ARC_semantics_calculation.py
     python ARH_semantics_calculation.py
     ```
   - Each script will traverse the `benchmarks/` directory, parse `.bag` files, and iterate until convergence or a maximum iteration count is reached.

4. **Output**  
   - For each `.bag` file processed:
     - `*_iter.csv` (or `*_iterative.csv`) is generated, containing iteration-by-iteration \(\sigma^+(a)\) and \(\sigma^-(a)\).
     - `*_final.csv` contains the final (or last-iteration) values of \(\sigma^+(a)\) and \(\sigma^-(a)\).
   - By default, the threshold for convergence is `1e-4`, and the maximum iterations is 20. These can be changed by editing parameters (`epsilon`, `max_iterations`) in each script.

---

## Dependencies

- **Python 3.7+**  
- **pandas** (for CSV output):
  ```bash
  pip install pandas
