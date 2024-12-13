import json
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(    
    api_key=os.environ["OPENAI_API_KEY"]
)

# Input and output files
input_file = "data/CV_images_tinyllava-6-24-24-val.json"
output_dir = "output_batches_val"
final_output_file = f"{output_dir}/combined_output_val.json"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Constants
PROCESS_ALL = False
NUM_ENTRIES_TO_PROCESS = 10
BATCH_SIZE = 2  # Process how many entries per batch
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
            Provide concise answers for each question."""}
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
        entry["caption"] = caption
        entry["answers"] = answers
        processed_entries.append(entry)
        print(f"Processed {i}th entry")
    return processed_entries

if __name__ == "__main__":
    with open(input_file, "r") as file:
        data = json.load(file)

    total_entries = len(data) if PROCESS_ALL else NUM_ENTRIES_TO_PROCESS
    for batch_start in range(0, total_entries, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_entries)
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
