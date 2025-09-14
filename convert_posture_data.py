#!/usr/bin/env python3
"""
Convert live posture data to posture_scores.json format
Transforms posture_data_TIMESTAMP.json files to the format expected by the meditation pipeline
"""

import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

def convert_posture_data_to_scores(posture_data_file: str, output_file: str = None) -> Dict[str, Any]:
    """
    Convert live posture data to posture_scores.json format
    
    Args:
        posture_data_file: Path to posture_data_TIMESTAMP.json file
        output_file: Optional output file path (defaults to preprocess_output/posture_scores.json)
    
    Returns:
        Converted posture data in the expected format
    """
    
    # Load the live posture data
    with open(posture_data_file, 'r', encoding='utf-8') as f:
        live_data = json.load(f)
    
    # Extract key information
    session_info = live_data.get('session_info', {})
    posture_stats = live_data.get('posture_statistics', {})
    frame_data = live_data.get('frame_data', [])
    
    # Calculate derived metrics for debug information
    if frame_data:
        # Extract scores and metrics from frame data
        scores = [frame.get('posture_score', 0.0) for frame in frame_data if frame.get('landmarks_detected', False)]
        shoulder_scores = [frame.get('metrics', {}).get('shoulder_alignment_score', 0.0) for frame in frame_data if frame.get('landmarks_detected', False)]
        head_scores = [frame.get('metrics', {}).get('head_position_score', 0.0) for frame in frame_data if frame.get('landmarks_detected', False)]
        spine_scores = [frame.get('metrics', {}).get('spine_alignment_score', 0.0) for frame in frame_data if frame.get('landmarks_detected', False)]
        confidence_scores = [frame.get('metrics', {}).get('detection_confidence', 0.0) for frame in frame_data if frame.get('landmarks_detected', False)]
        
        # Calculate stability (variance of scores)
        score_stability = 1.0 - np.var(scores) if len(scores) > 1 else 1.0
        
        # Calculate average metrics
        avg_shoulder_score = np.mean(shoulder_scores) if shoulder_scores else 0.0
        avg_head_score = np.mean(head_scores) if head_scores else 0.0
        avg_spine_score = np.mean(spine_scores) if spine_scores else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Calculate angle score (based on shoulder angle stability)
        shoulder_angles = [frame.get('metrics', {}).get('shoulder_angle', 0.0) for frame in frame_data if frame.get('landmarks_detected', False)]
        angle_variance = np.var(shoulder_angles) if len(shoulder_angles) > 1 else 0.0
        angle_score = max(0.0, 1.0 - (angle_variance / 100.0))  # Normalize angle variance
        
        # Calculate slope score (based on shoulder slope consistency)
        shoulder_slopes = [frame.get('metrics', {}).get('shoulder_slope', 0.0) for frame in frame_data if frame.get('landmarks_detected', False)]
        slope_variance = np.var(shoulder_slopes) if len(shoulder_slopes) > 1 else 0.0
        slope_score = max(0.0, 1.0 - (slope_variance * 1000.0))  # Normalize slope variance
        
    else:
        # Default values if no frame data
        score_stability = 0.0
        angle_score = 0.0
        slope_score = 0.0
        avg_confidence = 0.0
    
    # Create the converted data structure
    converted_data = {
        "file": f"live_posture_session_{session_info.get('timestamp', 'unknown')}",
        "posture_score": posture_stats.get('average_score', 0.0),
        "debug": {
            "angle_score": angle_score,
            "slope_score": slope_score,
            "stability_score": score_stability,
            "visibility_factor": avg_confidence,
            "detection_rate": posture_stats.get('detection_rate', 0.0),
            "session_duration": session_info.get('duration_seconds', 0),
            "total_frames": session_info.get('total_frames', 0),
            "detected_frames": len([f for f in frame_data if f.get('landmarks_detected', False)]),
            "max_score": posture_stats.get('max_score', 0.0),
            "min_score": posture_stats.get('min_score', 0.0),
            "session_quality": "excellent" if posture_stats.get('average_score', 0.0) >= 0.85 else
                             "good" if posture_stats.get('average_score', 0.0) >= 0.70 else
                             "fair" if posture_stats.get('average_score', 0.0) >= 0.55 else "poor"
        }
    }
    
    # Save to output file if specified
    if output_file:
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if file exists
        existing_data = []
        if output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        
        # Add new data
        existing_data.append(converted_data)
        
        # Save updated data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f"Converted posture data saved to: {output_file}")
    
    return converted_data

def convert_latest_posture_data(output_file: str = "preprocess_output/posture_scores.json") -> Dict[str, Any]:
    """
    Convert the most recent posture data file to posture_scores.json format
    
    Args:
        output_file: Output file path (defaults to preprocess_output/posture_scores.json)
    
    Returns:
        Converted posture data
    """
    
    # Find the most recent posture data file
    posture_files = glob.glob("preprocess_input/posture_data_*.json")
    
    if not posture_files:
        print("No posture data files found in preprocess_input/")
        return {}
    
    # Get the most recent file
    latest_file = max(posture_files, key=os.path.getctime)
    print(f"Converting latest posture data: {latest_file}")
    
    return convert_posture_data_to_scores(latest_file, output_file)

def convert_all_posture_data(output_file: str = "preprocess_output/posture_scores.json") -> List[Dict[str, Any]]:
    """
    Convert all posture data files to posture_scores.json format
    
    Args:
        output_file: Output file path (defaults to preprocess_output/posture_scores.json)
    
    Returns:
        List of converted posture data
    """
    
    # Find all posture data files
    posture_files = glob.glob("preprocess_input/posture_data_*.json")
    
    if not posture_files:
        print("No posture data files found in preprocess_input/")
        return []
    
    print(f"Found {len(posture_files)} posture data files")
    
    # Convert all files
    converted_data = []
    for posture_file in sorted(posture_files):
        print(f"Converting: {posture_file}")
        try:
            data = convert_posture_data_to_scores(posture_file)
            converted_data.append(data)
        except Exception as e:
            print(f"Error converting {posture_file}: {e}")
    
    # Save all converted data
    if converted_data:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"All converted posture data saved to: {output_file}")
    
    return converted_data

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert live posture data to posture_scores.json format")
    parser.add_argument("--input", type=str, help="Specific posture data file to convert")
    parser.add_argument("--output", type=str, default="preprocess_output/posture_scores.json", 
                       help="Output file path (default: preprocess_output/posture_scores.json)")
    parser.add_argument("--all", action="store_true", help="Convert all posture data files")
    parser.add_argument("--latest", action="store_true", help="Convert only the latest posture data file")
    
    args = parser.parse_args()
    
    if args.input:
        # Convert specific file
        if not os.path.exists(args.input):
            print(f"Error: File {args.input} not found")
            return 1
        
        print(f"Converting specific file: {args.input}")
        result = convert_posture_data_to_scores(args.input, args.output)
        print(f"Conversion completed. Result: {result}")
        
    elif args.all:
        # Convert all files
        print("Converting all posture data files...")
        results = convert_all_posture_data(args.output)
        print(f"Conversion completed. {len(results)} files processed.")
        
    elif args.latest:
        # Convert latest file
        print("Converting latest posture data file...")
        result = convert_latest_posture_data(args.output)
        print(f"Conversion completed. Result: {result}")
        
    else:
        # Default: convert latest file
        print("Converting latest posture data file (default behavior)...")
        result = convert_latest_posture_data(args.output)
        print(f"Conversion completed. Result: {result}")
    
    return 0

if __name__ == "__main__":
    exit(main())
