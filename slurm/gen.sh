#!/bin/bash

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <script> [partition_type]"
    exit 1
fi

python_file="$1"
partition_type="${2:-default}"
full_file_name="${python_file%.*}"

path_parts=( $(echo "$full_file_name" | tr '/' ' ') )
num_parts=${#path_parts[@]}

if [ $num_parts -ge 2 ]; then
    last_two_dirs="${path_parts[$num_parts-2]}/${path_parts[$num_parts-1]}"
else
    last_two_dirs="$full_file_name"
fi

base_file_name=$(basename "$full_file_name")

if [ "$partition_type" = "ica100" ]; then
    partition="ica100"
    mem="80G"
    time="12:00:00"
else
    partition="a100"
    mem="40G"
    time="8:00:00"
fi

cat > "${full_file_name}.slurm" << EOL
#!/bin/bash
#SBATCH --job-name=${last_two_dirs}
#SBATCH -A jeisner1_gpu
#SBATCH --partition=${partition}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=${full_file_name}.out

uv run ${python_file}
EOL

chmod +x "${full_file_name}.slurm"
echo "Generated ${full_file_name}.slurm"
