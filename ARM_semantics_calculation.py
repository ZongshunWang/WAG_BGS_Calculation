
# ARM_semantics_calculation.py
#
# Description:
#   This script computes the bilateral gradual semantics (AR-max-based, ARM)
#   for Weighted Argumentation Graphs (WAG). It reads .bag files containing
#   arguments (basic weights) and attacks, then performs iterative
#   updates of f(a) (acceptability) and g(a) (rejectability) until convergence
#   or a maximum number of iterations is reached.
#
# Usage:
#   1) Place .bag files under a directory named 'benchmarks/'.
#   2) Run: python ARM_semantics_calculation.py
#   3) The script will traverse the 'benchmarks/' folder, parse each .bag file,
#      and compute the final (f, g) degrees under ARM semantics, saving them to CSV.
#
# Formulas (ARM):
#   f(a) = w(a) / [ 1 + max_{b in Att(a)}( f(b)/(1+g(b)) ) ]
#   g(a) = [ max_{b in Att(a)} f(b) ] / [ 1 + max_{b in Att(a)} f(b) ]
#
# If Att(a) is empty, we treat the max term as 0.
#
# ------------------------------------------------------------------------------

import os
import pandas as pd

def compute_f_arm(arg, attacks_dict, f_prev, g_prev, weights):
    """
    Compute f(a) for argument 'arg' under ARM semantics:
      f(a) = w(a) / (1 + max_{b in Att(a)}( f(b)/(1 + g(b)) ))
    If there are no attackers, f(a) = w(a).
    """
    attackers = attacks_dict.get(arg, set())
    if not attackers:
        return weights[arg]
    else:
        infl = max(f_prev[b] / (1.0 + g_prev[b]) for b in attackers)
        return weights[arg] / (1.0 + infl)

def compute_g_arm(arg, attacks_dict, f_prev, g_prev):
    """
    Compute g(a) for argument 'arg' under ARM semantics:
      g(a) = [max_{b in Att(a)} f(b)] / [1 + max_{b in Att(a)} f(b)]
    If there are no attackers, g(a) = 0.
    """
    attackers = attacks_dict.get(arg, set())
    if not attackers:
        return 0.0
    else:
        max_f = max(f_prev[b] for b in attackers)
        return max_f / (1.0 + max_f)

def process_bag_file_ARM(bag_file_path, epsilon=1e-4, max_iterations=20):
    """
    Parse a single .bag file, run iterative ARM updates, and output final results.
    """
    weights = {}
    attacks_dict = {}

    # 1) Parse .bag file (only arg(...) / att(...))
    try:
        with open(bag_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith("arg(") and line.endswith(")"):
                    # Example: arg(a, 0.6)
                    content = line[4:-1]
                    arg_name, w_str = content.split(',')
                    arg_name = arg_name.strip()
                    w_val = float(w_str.strip())
                    weights[arg_name] = w_val
                elif line.startswith("att(") and line.endswith(")"):
                    # Example: att(x,a)
                    content = line[4:-1]
                    attacker, attacked = content.split(',')
                    attacker = attacker.strip()
                    attacked = attacked.strip()
                    if attacked not in attacks_dict:
                        attacks_dict[attacked] = set()
                    attacks_dict[attacked].add(attacker)
                else:
                    # Ignore lines not matching arg(...) or att(...)
                    continue
    except FileNotFoundError:
        print(f"Error: File {bag_file_path} not found.")
        return
    except Exception as e:
        print(f"Error processing file {bag_file_path}: {e}")
        return

    # Collect all argument names
    arguments = sorted(weights.keys())

    # 2) Initialize f^0(a)=w(a), g^0(a)=0
    f_current = {arg: weights[arg] for arg in arguments}
    g_current = {arg: 0.0 for arg in arguments}

    # 3) Iterative updates
    iteration = 0
    converged = False
    iterative_logs = []

    while iteration < max_iterations and not converged:
        iteration += 1
        f_next = {}
        g_next = {}
        max_delta = 0.0

        for arg in arguments:
            f_val = compute_f_arm(arg, attacks_dict, f_current, g_current, weights)
            g_val = compute_g_arm(arg, attacks_dict, f_current, g_current)
            f_next[arg] = f_val
            g_next[arg] = g_val

            delta_f = abs(f_val - f_current[arg])
            delta_g = abs(g_val - g_current[arg])
            max_delta = max(max_delta, delta_f, delta_g)

        iterative_logs.append((iteration, f_next.copy(), g_next.copy()))
        f_current = f_next
        g_current = g_next

        if max_delta < epsilon:
            converged = True
            print(f"[ARM] Converged after {iteration} iterations: {os.path.basename(bag_file_path)}.")

    if not converged:
        print(f"[ARM] Did not converge within {max_iterations} iterations: {os.path.basename(bag_file_path)}.")

    # 4) Print final results
    print(f"\n[ARM] Final degrees for {os.path.basename(bag_file_path)}:")
    print("---------------------------------------")
    print(f"{'Argument':<10} {'f(a)':<12} {'g(a)':<12}")
    for arg in arguments:
        print(f"{arg:<10} {f_current[arg]:<12.6f} {g_current[arg]:<12.6f}")

    # 5) Save iteration logs to CSV
    iter_data = []
    for (it_num, f_vals, g_vals) in iterative_logs:
        for arg in arguments:
            iter_data.append({
                'Iteration': it_num,
                'Argument': arg,
                'f(a)': f_vals[arg],
                'g(a)': g_vals[arg]
            })
    df_iter = pd.DataFrame(iter_data)
    csv_iter_path = os.path.splitext(bag_file_path)[0] + '_arm_iter.csv'
    df_iter.to_csv(csv_iter_path, index=False)

    # 6) Save final result to CSV
    final_df = pd.DataFrame({
        'Argument': arguments,
        'f(a)': [f_current[a] for a in arguments],
        'g(a)': [g_current[a] for a in arguments]
    })
    csv_final_path = os.path.splitext(bag_file_path)[0] + '_arm_final.csv'
    final_df.to_csv(csv_final_path, index=False)

def main():
    """
    Main entry: traverse the 'benchmarks/' directory, parse each .bag file,
    and compute ARM semantics results.
    """
    benchmarks_dir = os.path.join(os.getcwd(), 'benchmarks')
    if not os.path.exists(benchmarks_dir):
        print(f"Benchmarks directory not found: {benchmarks_dir}")
        return

    for root, dirs, files in os.walk(benchmarks_dir):
        for file_name in files:
            if file_name.endswith('.bag'):
                file_path = os.path.join(root, file_name)
                process_bag_file_ARM(file_path)

if __name__ == "__main__":
    main()
