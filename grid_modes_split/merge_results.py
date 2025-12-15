import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

def merge_and_extract_top_models():
    """
    Merge all Grid Search results from output_grid directory,
    save all sorted results, and extract top 20 models
    """
    
    print("=" * 70)
    print("MERGING ALL GRID SEARCH RESULTS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Input directory
    input_dir = Path("output_grid")
    
    # Check if directory exists
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist!")
        return
    
    # Find incremental files
    pattern = str(input_dir / "results_split_*_incremental.csv")
    files = sorted(glob.glob(pattern))
    
    print(f"Searching in: {input_dir}/")
    print(f"Pattern: results_split_*_incremental.csv\n")
    
    if not files:
        print("Error: No files found!")
        print("\nFiles available in directory:")
        for item in input_dir.iterdir():
            print(f"   - {item.name}")
        return
    
    print(f"Found {len(files)} files:\n")
    for f in files:
        print(f"   {Path(f).name}")
    
    # Read and merge
    print(f"\n{'='*70}")
    print("Loading data...")
    print("=" * 70)
    
    all_data = []
    
    for file_path in files:
        # Extract split number
        filename = Path(file_path).name
        split_num = int(filename.split("split_")[1].split("_")[0])
        
        # Read file
        df = pd.read_csv(file_path)
        
        # Filter successful models
        df_success = df[df['status'] == 'success'].copy()
        
        if len(df_success) > 0:
            df_success['split'] = split_num
            all_data.append(df_success)
            print(f"Split {split_num:02d}: {len(df_success):5d} successful models")
    
    if not all_data:
        print("\nError: No successful models found!")
        return
    
    # Merge all
    df_merged = pd.concat(all_data, ignore_index=True)
    
    # Sort by val_f1 descending
    df_sorted = df_merged.sort_values('val_f1', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print("Results Summary")
    print("=" * 70)
    print(f"Total successful models: {len(df_sorted):,}")
    print(f"Number of splits: {len(all_data)}")
    print(f"Best F1: {df_sorted['val_f1'].max():.4f}")
    print(f"Worst F1: {df_sorted['val_f1'].min():.4f}")
    print(f"Average F1: {df_sorted['val_f1'].mean():.4f}")
    
    # Create output directory
    output_dir = Path("output_merge")
    output_dir.mkdir(exist_ok=True)
    
    # Save all results
    all_results_file = output_dir / "all_results_sorted_by_f1.csv"
    df_sorted.to_csv(all_results_file, index=False)
    
    print(f"\n{'='*70}")
    print("Saving All Results")
    print("=" * 70)
    print(f"File saved: {all_results_file}")
    print(f"Total records: {len(df_sorted):,}")
    
    # Extract top 20
    df_top20 = df_sorted.head(20).copy()
    
    print(f"\n{'='*70}")
    print("TOP 20 MODELS")
    print("=" * 70)
    
    for idx, row in df_top20.iterrows():
        print(f"{idx+1:2d}. Split {int(row['split']):02d} | "
              f"ID {int(row['combination_id']):5d} | "
              f"F1: {row['val_f1']:.4f} | "
              f"Acc: {row['val_acc']:.4f}")
    
    # Save top 20
    top20_file = output_dir / "top_20_models.csv"
    df_top20.to_csv(top20_file, index=False)
    
    print(f"\n{'='*70}")
    print("Saving Top 20 Results")
    print("=" * 70)
    print(f"File saved: {top20_file}")
    print(f"Total records: {len(df_top20)}")
    
    # Top 20 Statistics
    print(f"\n{'='*70}")
    print("Top 20 Statistics")
    print("=" * 70)
    print(f"Best F1: {df_top20['val_f1'].max():.4f}")
    print(f"Worst F1 (in top 20): {df_top20['val_f1'].min():.4f}")
    print(f"Average F1: {df_top20['val_f1'].mean():.4f}")
    
    print(f"\n{'='*70}")
    print("Done!")
    print("=" * 70)

if __name__ == "__main__":
    merge_and_extract_top_models()

