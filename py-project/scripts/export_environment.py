import os
import subprocess

import yaml


def export_combined_environment(output_file="merged_environment.yml"):
    # Step 1: Get conda env export
    print("[*] Exporting conda environment...")
    result = subprocess.run(["conda", "env", "export"], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error exporting environment:", result.stderr)
        return

    env_data = yaml.safe_load(result.stdout)

    # Step 2: Get pip freeze output
    print("[*] Exporting pip dependencies...")
    pip_result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)

    if pip_result.returncode != 0:
        print("Error running pip freeze:", pip_result.stderr)
        return

    pip_packages = pip_result.stdout.strip().splitlines()

    # Step 3: Inject pip packages into YAML
    print("[*] Merging pip dependencies into environment YAML...")
    dependencies = env_data.get("dependencies", [])

    # Find if pip block already exists
    for i, dep in enumerate(dependencies):
        if isinstance(dep, dict) and "pip" in dep:
            dependencies[i]["pip"].extend(pip_packages)
            break
    else:
        # Add pip section
        dependencies.append({"pip": pip_packages})

    # Make sure pip is listed as a conda dependency
    if "pip" not in [d for d in dependencies if isinstance(d, str)]:
        dependencies.insert(0, "pip")

    # Step 4: Write merged environment to file
    print(f"[*] Writing merged environment to {output_file}")
    with open(output_file, "w") as f:
        yaml.dump(env_data, f, sort_keys=False)

    print("[✔] Done!")


if __name__ == "__main__":
    export_combined_environment("environment.yml")
