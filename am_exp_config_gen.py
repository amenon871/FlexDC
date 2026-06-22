from pathlib import Path

BASE_EXPERIMENT_CONFIG = Path("configs/experiment/exp.ini")
OUTPUT_DIR = Path("configs/experiment/generated_server_counts")

SERVER_COUNTS = [250]

#THIS FILE IS USED TO GENERATE .ini files with different server counts

def replace_ini_value(text: str, key: str, new_value: str) -> str:
    lines = text.splitlines()
    new_lines = []

    replaced = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(f"{key}") and "=" in stripped and not stripped.startswith(";"):
            prefix = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{prefix}{key} = {new_value}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise ValueError(f"Could not find key '{key}' in {BASE_EXPERIMENT_CONFIG}")

    return "\n".join(new_lines) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_text = BASE_EXPERIMENT_CONFIG.read_text()

    for server_count in SERVER_COUNTS:
        edited = replace_ini_value(base_text, "server_count", str(server_count))

        output_path = OUTPUT_DIR / f"exp_servers_{server_count}.ini"
        output_path.write_text(edited)

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()