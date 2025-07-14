#!/bin/bash

# Bash script to clean pixel data from results.json files in directories p11 through p19
# Usage: ./run_clean_batch.sh

# Base directory path
BASE_DIR="$HOME/mount-folder/inference_output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting batch cleaning of pixel data for directories p11-p19${NC}"
echo "Base directory: $BASE_DIR"
echo "=================================="

# Track statistics
total_dirs=0
processed_dirs=0
failed_dirs=0
start_time=$(date +%s)

# Loop through directories p11 to p19
for i in {11..19}; do
    dir_name="p$i"
    full_path="$BASE_DIR/$dir_name"
    total_dirs=$((total_dirs + 1))

    echo ""
    echo -e "${YELLOW}Processing directory: $dir_name${NC}"
    echo "Path: $full_path"

    # Check if directory exists
    if [ ! -d "$full_path" ]; then
        echo -e "${RED}❌ Directory $full_path does not exist - skipping${NC}"
        failed_dirs=$((failed_dirs + 1))
        continue
    fi

    # Run the Python script
    echo "Running: python3 clean_pixel_data.py --input $full_path"

    if python3 clean_pixel_data.py --input "$full_path"; then
        echo -e "${GREEN}✅ Successfully processed $dir_name${NC}"
        processed_dirs=$((processed_dirs + 1))
    else
        echo -e "${RED}❌ Failed to process $dir_name${NC}"
        failed_dirs=$((failed_dirs + 1))
    fi
done

# Calculate elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
seconds=$((elapsed % 60))

echo ""
echo "=================================="
echo -e "${BLUE}Batch processing complete!${NC}"
echo "Summary:"
echo "  Total directories: $total_dirs"
echo -e "  Successfully processed: ${GREEN}$processed_dirs${NC}"
echo -e "  Failed: ${RED}$failed_dirs${NC}"
echo "  Time elapsed: ${minutes}m ${seconds}s"

if [ $failed_dirs -eq 0 ]; then
    echo -e "${GREEN}🎉 All directories processed successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some directories failed to process${NC}"
    exit 1
fi