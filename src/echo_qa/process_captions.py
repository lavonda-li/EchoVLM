import json
import os
import argparse
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(    
    api_key=os.environ["OPENAI_API_KEY"]
)

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="Process captions using OpenAI API.")
    parser.add_argument("--data_str", type=str, choices=["train", "val", "test"], help="Dataset type", required=True)
    parser.add_argument("--input_dir", type=str, default="data", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="output_batches", help="Output directory")
    parser.add_argument("--process_all", action="store_true", default=True, help="Process all entries")
    parser.add_argument("--num_entries_to_process", type=int, default=10, help="Number of entries to process if not processing all")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of entries to process per batch")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing")
    return parser

# Constants
QUESTIONS_LIST = [
    "Q1: What imaging modality is represented in this image?",
    "Q2: What body region or anatomical area does this image depict?",
    "Q3: Are there any abnormalities identified in this image?",
    "Q4: Does this image appear normal, or does it show any irregularities?",
    "Q5: Does this image contain any label or index that is significant or noteworthy?",
]

def process_caption(caption):
    """Send a caption to the OpenAI API for processing."""
    question_str = "\n".join(QUESTIONS_LIST)
    messages = [
        {"role": "system", "content": "You are a medical expert trained to interpret medical image captions."},
        {"role": "user", "content": f"""
            For the provided caption, answer the following questions strictly based on the caption:
            Caption: {caption}
            {question_str}
            Provide concise answers for each question. For each answer, start with 'A1: ' for answer 1 and so on."""}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip().split("\n")

def save_to_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def process_data(data, start_idx, end_idx):
    """Process a slice of data entries."""
    processed_entries = []
    for i in range(start_idx, end_idx):
        entry = data[i]
        caption = entry["conversations"][-1]["value"].strip()  # Extract GPT caption
        answers = process_caption(caption)
        
        # Update entry with processed answers
        entry.pop("conversations", None)
        entry["caption"] = caption
        entry["answers"] = list(answers)
        processed_entries.append(entry)
        print(f"Processed {i}th entry")
    return processed_entries

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    # print important arguments
    print(f"Data string: {args.data_str}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Process all: {args.process_all}")
    print(f"Number of entries to process: {args.num_entries_to_process}")
    print(f"Batch size: {args.batch_size}")
    print(f"Start index: {args.start_idx}")
    

    # Input and output files
    input_file = os.path.join(args.input_dir, f"CV_images_tinyllava-6-24-24-{args.data_str}.json")
    output_dir = f"{args.output_dir}_{args.data_str}"
    final_output_file = os.path.join(output_dir, f"combined_output_{args.data_str}.json")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as file:
        data = json.load(file)

    print(f"Loaded {len(data)} entries from '{input_file}'")
    total_entries = len(data) if args.process_all else args.num_entries_to_process
    for batch_start in range(args.start_idx, total_entries, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total_entries)
        print(f"Processing batch {batch_start} to {batch_end - 1}")
        batch_data = process_data(data, batch_start, batch_end)
        
        # Save each batch to a separate file
        batch_file = os.path.join(output_dir, f"batch_{batch_start}_{batch_end - 1}.json")
        save_to_json(batch_data, batch_file)
        print(f"Batch saved to {batch_file}")

    # Combine all batch files into one
    combined_data = []
    for batch_file in sorted(os.listdir(output_dir)):
        if batch_file.endswith(".json"):
            with open(os.path.join(output_dir, batch_file), "r") as f:
                combined_data.extend(json.load(f))
    save_to_json(combined_data, final_output_file)
    print(f"All batches combined into '{final_output_file}'.")