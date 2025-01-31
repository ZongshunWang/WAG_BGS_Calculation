
# ARH_semantics_calculation.py
#
# Description:
#   This script computes the bilateral gradual semantics (AR-hybrid-based, ARH)
#   for Weighted Argumentation Graphs (WAG). It reads .bag files with
#   arguments (intrinsic weights) and attacks, then iterates f(a), g(a) until
#   convergence or a maximum number of iterations.
#
# Usage:
#   1) Place .bag files under 'benchmarks/'.
#   2) Run: python ARH_semantics_calculation.py
#   3) The script will parse each .bag file and compute final (f, g) under
#      ARH semantics, saving to CSV.
#
# Formulas (ARH):
#   Let Att^*(a) = { b | w(b)>0 and b attacks a }.
#   f(a) = w(a) / [ 1 + |Att^*(a)| + sum_{b in Att^*(a)}( f(b)/(1+g(b)) ) ]
#   g(a) = [ |Att^*(a)| + sum_{b in Att^*(a)} f(b ) ] / [ 1 + ... ]
#
# If Att^*(a) is empty, then f(a)=w(a), g(a)=0.
#
# ------------------------------------------------------------------------------

import os
import pandas as pd

def compute_f_arh(arg, attacks_dict, f_prev, g_prev, weights):
    """
    Compute f(a) under ARH semantics:
      f(a) = w(a) / [1 + |Att^*(a)| + sum(f(b)/(1+g(b)) )]
    """
    attackers = attacks_dict.get(arg, set())
    founded_atts = [b for b in attackers if weights.get(b, 0.0) > 0]
    if not founded_atts:
        return weights[arg]

    sum_infl = sum(f_prev[b] / (1.0 + g_prev[b]) for b in founded_atts)
    return weights[arg] / (1.0 + len(founded_atts) + sum_infl)

def compute_g_arh(arg, attacks_dict, f_prev, g_prev, weights):
    """
    Compute g(a) under ARH semantics:
      g(a) = [ |Att^*(a)| + sum(f(b)) ] / [ 1 + ... ]
    """
    attackers = attacks_dict.get(arg, set())
    founded_atts = [b for b in attackers if weights.get(b, 0.0) > 0]
    if not founded_atts:
        return 0.0

    sum_f = sum(f_prev[b] for b in founded_atts)
    numerator = len(founded_atts) + sum_f
    denominator = 1.0 + len(founded_atts) + sum_f
    return numerator / denominator

def process_bag_file_ARH(bag_file_path, epsilon=1e-4, max_iterations=20):
    """
    Parse the .bag file, run ARH iterative updates, and output the final degrees.
    """
    weights = {}
    attacks_dict = {}

    # 1) Parse .bag
    try:
        with open(bag_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith("arg(") and line.endswith(")"):
                    content = line[4:-1]
                    arg_name, w_str = content.split(',')
                    arg_name = arg_name.strip()
                    w_val = float(w_str.strip())
                    weights[arg_name] = w_val
                elif line.startswith("att(") and line.endswith(")"):
                    content = line[4:-1]
                    attacker, attacked = content.split(',')
                    attacker = attacker.strip()
                    attacked = attacked.strip()
                    if attacked not in attacks_dict:
                        attacks_dict[attacked] = set()
                    attacks_dict[attacked].add(attacker)
                else:
                    continue
    except Exception as e:
        print(f"[ARH] Error reading file {bag_file_path}: {e}")
        return

    arguments = sorted(weights.keys())
    # Initialize
    f_current = {a: weights[a] for a in arguments}
    g_current = {a: 0.0 for a in arguments}

    iteration = 0
    converged = False
    iterative_logs = []

    while iteration < max_iterations and not converged:
        iteration += 1
        f_next = {}
        g_next = {}
        max_delta = 0.0

        for a in arguments:
            f_val = compute_f_arh(a, attacks_dict, f_current, g_current, weights)
            g_val = compute_g_arh(a, attacks_dict, f_current, g_current, weights)
            f_next[a] = f_val
            g_next[a] = g_val

            df = abs(f_val - f_current[a])
            dg = abs(g_val - g_current[a])
            max_delta = max(max_delta, df, dg)

        iterative_logs.append((iteration, f_next.copy(), g_next.copy()))
        f_current = f_next
        g_current = g_next

        if max_delta < epsilon:
            converged = True
            print(f"[ARH] Converged after {iteration} iterations: {os.path.basename(bag_file_path)}.")

    if not converged:
        print(f"[ARH] Not converged within {max_iterations} iterations: {os.path.basename(bag_file_path)}.")

    # Print final
    print(f"\n[ARH] Final degrees for {os.path.basename(bag_file_path)}:")
    print("---------------------------------------")
    print(f"{'Argument':<10} {'f(a)':<12} {'g(a)':<12}")
    for a in arguments:
        print(f"{a:<10} {f_current[a]:<12.6f} {g_current[a]:<12.6f}")

    # CSV logs
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
    csv_iter_path = os.path.splitext(bag_file_path)[0] + '_arh_iter.csv'
    df_iter.to_csv(csv_iter_path, index=False)

    # Final CSV
    final_df = pd.DataFrame({
        'Argument': arguments,
        'f(a)': [f_current[a] for a in arguments],
        'g(a)': [g_current[a] for a in arguments]
    })
    csv_final_path = os.path.splitext(bag_file_path)[0] + '_arh_final.csv'
    final_df.to_csv(csv_final_path, index=False)

def main():
    """
    Main entry: traverse 'benchmarks/' for .bag files and compute ARH semantics.
    """
    benchmarks_dir = os.path.join(os.getcwd(), 'benchmarks')
    if not os.path.exists(benchmarks_dir):
        print(f"[ARH] Benchmarks directory not found: {benchmarks_dir}")
        return

    for root, dirs, files in os.walk(benchmarks_dir):
        for file_name in files:
            if file_name.endswith('.bag'):
                file_path = os.path.join(root, file_name)
                process_bag_file_ARH(file_path)

if __name__ == "__main__":
    main()
