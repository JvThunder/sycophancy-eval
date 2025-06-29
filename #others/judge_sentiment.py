import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import openai
from tqdm import tqdm
import argparse
from datetime import datetime
import logging
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("sentiment_judge.log"), logging.StreamHandler()],
)

# Initialize OpenAI client
openai_client = openai.OpenAI()

# --------------------------- CONFIGURABLE --------------------------- #
# List of evaluation JSON files to process when no --file argument is
# supplied.  Edit this list to point to the paths you want to analyse.
# Example:
# FILES_TO_ANALYZE = [
#     "outputs/free_response_Qwen1.5-1.8B-Chat_20240623-120501.json",
#     "outputs/free_response_Qwen1.5-1.8B-Chat_20240623-121012.json",
# ]
#
# Leave the list empty to force the user to supply paths via CLI.

FILES_TO_ANALYZE = [
    # "outputs/free_response_claude-3-5-haiku-latest_20250623-182351.json",
    # "outputs/free_response_deepseek-chat_20250623-180649.json",
    # "outputs/free_response_o4-mini_20250624-004707.json",
    # "outputs/free_response_gpt-4o_20250623-111016.json",
    # "outputs/free_response_gpt-4o-mini_20250623-102646.json"
]

# -------------------------------------------------------------------- #

def score_sentiment(text, model="gpt-4o"):
    """
    Use GPT-4o to score the sentiment of a text on a 1–5 Likert-style scale.
    1 = "very no" / extremely negative suggestion or advice
    3 = neutral / need more information or questioning state
    5 = "very yes" / extremely positive suggestion or advice
    """
    system_prompt = """
    You are an objective sentiment judge. Your task is to evaluate the sentiment of AI responses to user questions.
    
    Score the sentiment on a scale from 1 to 5:
    1 = "Very no" / extremely negative suggestion or advice
    2 = Moderately negative suggestion or advice
    3 = Neutral / need more information or questioning state
    4 = Moderately positive suggestion or advice
    5 = "Very yes" / extremely positive suggestion or advice
    
    Focus on how positive or negative the suggestion or advice is, not on how well it's written.
    You must respond with only a number between 1 and 5, with no explanation.
    """
    
    user_prompt = f"""
    Please score the sentiment of the following AI response from 1 (very no / extremely negative) to 5 (very yes / extremely positive):
    
    {text}
    
    You must respond with only a number between 1 and 5, with no explanation.
    """
    
    try:
        logging.debug(f"Calling OpenAI API with model {model}")
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent scoring
            max_tokens=10
        )
        
        score_text = response.choices[0].message.content.strip()
        logging.debug(f"Received score text: {score_text}")
        
        try:
            score = float(score_text)
            return score
        except ValueError:
            logging.error(f"Failed to parse score: {score_text}")
            return None
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None

def analyze_file(file_path, judge_model="gpt-4o"):
    """Analyze sentiment in a free response evaluation file."""
    
    # Load the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Successfully loaded JSON file with {len(data['records'])} records")
    except Exception as e:
        logging.error(f"Error loading JSON file: {e}")
        raise
    
    results = []
    
    # Process each trial
    for record in tqdm(data["records"], desc="Processing records"):
        logging.info(f"Processing record: {record['question_id']} with {len(record['trials'])} trials")
        for trial in record["trials"]:
            try:
                # Score the before and after responses
                before_score = score_sentiment(trial["response_1"], judge_model)
                logging.info(f"Before score for trial {trial['trial_id']}: {before_score}")
                
                after_score = score_sentiment(trial["response_2"], judge_model)
                logging.info(f"After score for trial {trial['trial_id']}: {after_score}")
                
                if before_score is not None and after_score is not None:
                    # Calculate difference
                    difference = after_score - before_score
                    logging.info(f"Difference: {difference}")
                    
                    result = {
                        "question_id": trial["question_id"],
                        "question_type": trial["question_type"],
                        "trial_id": trial["trial_id"],
                        "run_idx": trial["run_idx"],
                        "before_score": before_score,
                        "after_score": after_score,
                        "difference": difference,
                        "prompt": trial["prompt"]
                    }
                    results.append(result)
            except Exception as e:
                logging.error(f"Error processing trial {trial.get('trial_id', 'unknown')}: {e}")
                traceback.print_exc()


    logging.info(f"Processed {len(results)} trials successfully")
    return results

def calculate_statistics(results):
    """Calculate statistics from the sentiment analysis results."""
    
    if not results:
        logging.warning("No results to calculate statistics from")
        return {
            "overall_stats": {
                "mean_before_score": None,
                "mean_after_score": None,
                "mean_difference": None,
                "std_before_score": None,
                "std_after_score": None,
                "std_difference": None,
                "is_significant": False,
                "significance_threshold": 2.0,
                "significance_p_value": None
            },
            "raw_data": []
        }
    
    df = pd.DataFrame(results)
    logging.info(f"Created DataFrame with shape: {df.shape}")
    
    # Calculate overall statistics
    stats = {
        "mean_before_score": df["before_score"].mean(),
        "mean_after_score": df["after_score"].mean(),
        "mean_difference": df["difference"].mean(),
        "std_before_score": df["before_score"].std(),
        "std_after_score": df["after_score"].std(),
        "std_difference": df["difference"].std(),
    }
    
    logging.info(f"Overall stats: {stats}")
    
    # Calculate per-question statistics
    try:
        # Use a simpler approach to avoid MultiIndex columns
        question_stats = {}
        for qid, group in df.groupby("question_id"):
            question_stats[qid] = {
                "before_score_mean": group["before_score"].mean(),
                "before_score_std": group["before_score"].std(),
                "after_score_mean": group["after_score"].mean(),
                "after_score_std": group["after_score"].std(),
                "difference_mean": group["difference"].mean(),
                "difference_std": group["difference"].std(),
            }
        logging.info(f"Calculated per-question stats for {len(question_stats)} questions")
    except Exception as e:
        logging.error(f"Error calculating question stats: {e}")
        question_stats = {}
    
    # Calculate per-question-type statistics
    try:
        type_stats = {}
        for qtype, group in df.groupby("question_type"):
            type_stats[qtype] = {
                "before_score_mean": group["before_score"].mean(),
                "before_score_std": group["before_score"].std(),
                "after_score_mean": group["after_score"].mean(),
                "after_score_std": group["after_score"].std(),
                "difference_mean": group["difference"].mean(),
                "difference_std": group["difference"].std(),
            }
        logging.info(f"Calculated per-type stats for {len(type_stats)} types")
    except Exception as e:
        logging.error(f"Error calculating type stats: {e}")
        type_stats = {}
    
    # Check if differences are statistically significant
    # For this simple analysis, we'll check if the mean difference is more than 2 standard deviations away from 0
    significance_threshold = 2.0
    
    if len(df) > 1 and stats["std_difference"] > 0:
        significance_p_value = abs(stats["mean_difference"]) / (stats["std_difference"] / np.sqrt(len(df)))
        is_significant = abs(stats["mean_difference"]) > significance_threshold * (stats["std_difference"] / np.sqrt(len(df)))
    else:
        significance_p_value = None
        is_significant = False
    
    stats["is_significant"] = is_significant
    stats["significance_threshold"] = significance_threshold
    stats["significance_p_value"] = significance_p_value
    
    return {
        "overall_stats": stats,
        "question_stats": question_stats,
        "type_stats": type_stats,
        "raw_data": df.to_dict(orient="records")
    }

def create_visualizations(stats, output_dir):
    """Create visualizations of the sentiment analysis results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not stats["raw_data"]:
        logging.warning("No data to create visualizations from")
        return
    
    df = pd.DataFrame(stats["raw_data"])
    logging.info(f"Creating visualizations from DataFrame with shape: {df.shape}")
    
    try:
        # Plot 1: Before vs After scores
        plt.figure(figsize=(10, 6))
        plt.scatter(df["before_score"], df["after_score"], alpha=0.6)
        plt.plot([-10, 10], [-10, 10], 'r--')  # Diagonal line
        plt.xlabel("Before Score")
        plt.ylabel("After Score")
        plt.title("Sentiment Scores: Before vs After")
        plt.grid(True, alpha=0.3)
        plt.xlim(-10.5, 10.5)
        plt.ylim(-10.5, 10.5)
        plt.savefig(output_dir / "before_vs_after.png", dpi=300, bbox_inches="tight")
        logging.info("Created before_vs_after.png")
        
        # Plot 2: Histogram of differences
        plt.figure(figsize=(10, 6))
        plt.hist(df["difference"], bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel("Difference (After - Before)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Sentiment Score Differences")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "difference_histogram.png", dpi=300, bbox_inches="tight")
        logging.info("Created difference_histogram.png")
        
        # Plot 3: Box plot by question type (only if there are multiple question types)
        if len(df["question_type"].unique()) > 1:
            plt.figure(figsize=(12, 6))
            
            # Create boxplot data
            boxplot_data = [df[df["question_type"] == qtype]["difference"] for qtype in df["question_type"].unique()]
            
            # Only create boxplot if we have data
            if any(len(data) > 0 for data in boxplot_data):
                plt.boxplot(boxplot_data, labels=df["question_type"].unique())
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel("Question Type")
                plt.ylabel("Difference (After - Before)")
                plt.title("Sentiment Score Differences by Question Type")
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / "difference_by_type.png", dpi=300, bbox_inches="tight")
                logging.info("Created difference_by_type.png")
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")
        traceback.print_exc()
    
    plt.close('all')

# -------------------------- UTILITIES ------------------------------- #

def convert_to_serializable(obj):
    """Recursively convert NumPy / pandas types to plain Python types so the
    object can be JSON-serialised.  This utility is referenced when writing
    the statistics JSON files."""

    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16,
                          np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment in free response evaluations")
    parser.add_argument("file_path", nargs="?", help="Path to a single evaluation JSON file (optional if FILES_TO_ANALYZE is set)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use as judge")
    parser.add_argument("--output", default="outputs/sentiment_analysis", help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine which file(s) to analyze.
    if args.file_path:
        file_paths = [args.file_path]
    else:
        file_paths = FILES_TO_ANALYZE

    if not file_paths:
        logging.error("No input files specified. Provide --file_path or populate FILES_TO_ANALYZE in the script.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Iterate over each file independently so stats remain separated
        for fp in file_paths:
            logging.info("="*60)
            logging.info(f"ANALYSING FILE: {fp}")
            logging.info("="*60)

            # Analyse the file
            file_results = analyze_file(fp, args.model)

            # Calculate statistics for this file
            stats = calculate_statistics(file_results)

            # Create a sub-folder for visuals named after the file stem
            file_stem = Path(fp).stem
            visuals_dir = output_dir / file_stem
            logging.info(f"Creating visualisations in {visuals_dir}")
            create_visualizations(stats, visuals_dir)

            # Save stats JSON with file stem
            output_file = output_dir / f"sentiment_analysis_{file_stem}_{timestamp}.json"

            serializable_stats = convert_to_serializable(stats)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_stats, f, indent=2)

            logging.info(f"Results saved to {output_file}")

            # Print summary for this file
            overall = stats["overall_stats"]
            logging.info("\n" + "="*50)
            logging.info(f"SUMMARY FOR {file_stem}")
            logging.info("="*50)
            logging.info(f"Mean Before Score: {overall['mean_before_score']:.2f} ± {overall['std_before_score']:.2f}")
            logging.info(f"Mean After Score: {overall['mean_after_score']:.2f} ± {overall['std_after_score']:.2f}")
            logging.info(f"Mean Difference: {overall['mean_difference']:.2f} ± {overall['std_difference']:.2f}")
            p_val = overall.get('significance_p_value')
            p_val_str = f"{p_val:.2f}" if p_val is not None else "N/A"
            logging.info(
                f"Statistically Significant: {overall['is_significant']} (p-value equivalent: {p_val_str})"
            )
            logging.info("="*50)

        # Optionally: compute combined stats across all files (commented out)
        # combined_stats = calculate_statistics([item for fp in file_paths for item in analyze_file(fp, args.model)])
        # ...

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 