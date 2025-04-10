#!/usr/bin/env python3
import sys
import numpy as np
import laspy
import os

def remove_duplicates(las_file_path):
    """
    Remove duplicate points (points with identical x, y, z coordinates) from a LAS file
    and save the result back to the same file path.
    
    Args:
        las_file_path (str): Path to the LAS file
    """
    print(f"Processing file: {las_file_path}")
    print("Loading LAS file...")
    
    # Read the input LAS file
    try:
        infile = laspy.file.File(las_file_path, mode="r")
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        return
    
    # Get the point data
    point_records = np.vstack((infile.x, infile.y, infile.z)).transpose()
    
    # Count original points
    original_count = len(point_records)
    print(f"Original point count: {original_count}")
    
    # Find unique points
    print("Removing duplicates...")
    unique_points, unique_indices = np.unique(point_records, axis=0, return_index=True)
    
    # Count unique points
    unique_count = len(unique_indices)
    duplicates_removed = original_count - unique_count
    print(f"Unique point count: {unique_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    
    # Create temporary output file
    temp_file_path = las_file_path + ".temp"
    
    try:
        # Create output file with same header as input
        outfile = laspy.file.File(temp_file_path, mode="w", header=infile.header)
        
        # Filter points by unique indices
        outfile.X = infile.X[unique_indices]
        outfile.Y = infile.Y[unique_indices]
        outfile.Z = infile.Z[unique_indices]
        
        # Copy any available point attributes
        if hasattr(infile, 'intensity'):
            outfile.intensity = infile.intensity[unique_indices]
        if hasattr(infile, 'return_number'):
            outfile.return_number = infile.return_number[unique_indices]
        if hasattr(infile, 'number_of_returns'):
            outfile.number_of_returns = infile.number_of_returns[unique_indices]
        if hasattr(infile, 'classification'):
            outfile.classification = infile.classification[unique_indices]
        if hasattr(infile, 'scan_angle_rank'):
            outfile.scan_angle_rank = infile.scan_angle_rank[unique_indices]
        if hasattr(infile, 'user_data'):
            outfile.user_data = infile.user_data[unique_indices]
        if hasattr(infile, 'point_source_id'):
            outfile.point_source_id = infile.point_source_id[unique_indices]
        
        # Close both files
        infile.close()
        outfile.close()
        
        # Replace original file with the temporary file
        os.replace(temp_file_path, las_file_path)
        print("File successfully saved!")
        
    except Exception as e:
        print(f"Error processing LAS file: {e}")
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_duplicates.py <path_to_las_file>")
        sys.exit(1)
    
    las_file_path = sys.argv[1]
    remove_duplicates(las_file_path) 