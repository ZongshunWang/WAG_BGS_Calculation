
# ARC_semantics_calculation.py
#
# Description:
#   This script computes the bilateral gradual semantics (AR-card-based, ARC)
#   for Weighted Argumentation Graphs (WAG). It reads .bag files containing
#   arguments (with intrinsic weights) and attacks, then performs iterative
#   updates until convergence or a maximum number of iterations.
#
# Usage:
#   1) Place .bag files under 'benchmarks/'.
#   2) Run: python ARC_semantics_calculation.py
#   3) The script will parse each .bag file and compute final (f, g) degrees
#      under ARC semantics, saving results to CSV.
#
# Formulas (ARC):
#   Let n = number of arguments (|A|).
#   Let Att^*(a) = { b | b attacks a and w(b) > 0 }.
#
#   f(a) = w(a) / [ 1 + |Att^*(a)| + (1/n)* sum_{b in Att^*(a)}( f(b)/(1+g(b)) ) ]
#   g(a) = [ |Att^*(a)| + (1/n)* sum_{b in Att^*(a)} f(b) ]
#          / [ 1 + |Att^*(a)| + (1/n)* sum_{b in Att^*(a)} f(b) ]
#
# If Att^*(a) is empty, f(a) = w(a), g(a)=0.
#
# ------------------------------------------------------------------------------

import os
import pandas as pd

def compute_f_arc(arg, attacks_dict, f_prev, g_prev, weights, arguments):
    """
    Compute f(a) under ARC semantics:
      f(a) = w(a) / [1 + |Att^*(a)| + (1/n)* sum( f(b)/(1+g(b)) ) ]
    where Att^*(a) includes attackers b with w(b)>0.
    """
    attackers = attacks_dict.get(arg, set())
    founded_atts = [b for b in attackers if weights.get(b, 0.0) > 0.0]
    if not founded_atts:
        return weights[arg]

    n = len(arguments)
    sum_infl = sum(f_prev[b] / (1.0 + g_prev[b]) for b in founded_atts)
    return weights[arg] / (1.0 + len(founded_atts) + (1.0/n)*sum_infl)

def compute_g_arc(arg, attacks_dict, f_prev, g_prev, weights, arguments):
    """
    Compute g(a) under ARC semantics:
      g(a) = [ |Att^*(a)| + (1/n)* sum_{b in Att^*(a)} f(b ) ] / [ 1 + ... ]
    If Att^*(a) is empty, g(a) = 0.
    """
    attackers = attacks_dict.get(arg, set())
    founded_atts = [b for b in attackers if weights.get(b, 0.0) > 0.0]
    if not founded_atts:
        return 0.0

    n = len(arguments)
    sum_f = sum(f_prev[b] for b in founded_atts)
    numerator = len(founded_atts) + (1.0/n)*sum_f
    denominator = 1.0 + len(founded_atts) + (1.0/n)*sum_f
    return numerator / denominator

def process_bag_file_ARC(bag_file_path, epsilon=1e-4, max_iterations=20):
    """
    Parse the .bag file, run ARC iterative updates, and output the results.
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
        print(f"[ARC] Error reading file {bag_file_path}: {e}")
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
            f_val = compute_f_arc(a, attacks_dict, f_current, g_current, weights, arguments)
            g_val = compute_g_arc(a, attacks_dict, f_current, g_current, weights, arguments)
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
            print(f"[ARC] Converged after {iteration} iterations: {os.path.basename(bag_file_path)}.")

    if not converged:
        print(f"[ARC] Not converged within {max_iterations} iterations: {os.path.basename(bag_file_path)}.")

    # Print final
    print(f"\n[ARC] Final degrees for {os.path.basename(bag_file_path)}:")
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
    csv_iter_path = os.path.splitext(bag_file_path)[0] + '_arc_iter.csv'
    df_iter.to_csv(csv_iter_path, index=False)

    # Final CSV
    final_df = pd.DataFrame({
        'Argument': arguments,
        'f(a)': [f_current[a] for a in arguments],
        'g(a)': [g_current[a] for a in arguments]
    })
    csv_final_path = os.path.splitext(bag_file_path)[0] + '_arc_final.csv'
    final_df.to_csv(csv_final_path, index=False)

def main():
    """
    Main entry: traverse 'benchmarks/' for .bag files and compute ARC semantics.
    """
    benchmarks_dir = os.path.join(os.getcwd(), 'benchmarks')
    if not os.path.exists(benchmarks_dir):
        print(f"[ARC] Benchmarks directory not found: {benchmarks_dir}")
        return

    for root, dirs, files in os.walk(benchmarks_dir):
        for file_name in files:
            if file_name.endswith('.bag'):
                file_path = os.path.join(root, file_name)
                process_bag_file_ARC(file_path)

if __name__ == "__main__":
    main()
