envs=("satimage-env" "internimage-env" "mmseg-env" "dinov2-env")
output_dir="$(dirname "$(dirname "$(realpath "$0")")")/env_requirements"
mkdir -p "$output_dir"

for env in "${envs[@]}"; do
    echo "Exporting requirements for $env..."
    conda run -n "$env" pip freeze > "$output_dir/$env.txt"
done

echo "All requirements exported to $output_dir/"
