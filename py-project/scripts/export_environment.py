import shutil
import subprocess  # nosec B404

import yaml


def export_combined_environment(output_file="merged_environment.yml"):
    print("[*] Exporting conda environment...")

    conda_exe = shutil.which("conda")
    if not conda_exe:
        print("Error: conda not found in PATH.")
        return

    result = subprocess.run(
        [conda_exe, "env", "export"], capture_output=True, text=True  # nosec B603
    )
    if result.returncode != 0:
        print("Error exporting environment:", result.stderr)
        return

    env_data = yaml.safe_load(result.stdout)

    print("[*] Exporting pip dependencies...")

    pip_exe = shutil.which("pip")
    if not pip_exe:
        print("Error: pip not found in PATH.")
        return

    pip_result = subprocess.run(
        [pip_exe, "freeze"], capture_output=True, text=True  # nosec B603
    )
    if pip_result.returncode != 0:
        print("Error running pip freeze:", pip_result.stderr)
        return

    pip_packages = pip_result.stdout.strip().splitlines()

    print("[*] Merging pip dependencies into environment YAML...")
    dependencies = env_data.get("dependencies", [])

    for i, dep in enumerate(dependencies):
        if isinstance(dep, dict) and "pip" in dep:
            dependencies[i]["pip"].extend(pip_packages)
            break
    else:
        dependencies.append({"pip": pip_packages})

    if "pip" not in [d for d in dependencies if isinstance(d, str)]:
        dependencies.insert(0, "pip")

    print(f"[*] Writing merged environment to {output_file}")
    with open(output_file, "w") as f:
        yaml.safe_dump(env_data, f, sort_keys=False)

    print("[✔] Done!")


if __name__ == "__main__":
    export_combined_environment("environment.yml")
