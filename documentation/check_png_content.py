#!/usr/bin/env python3
import sys
import os
from PIL import Image
import numpy as np

def analyze_png(filename):
    """Analyze PNG file content"""
    if not os.path.exists(filename):
        print(f"❌ {filename} does not exist")
        return
    
    try:
        img = Image.open(filename)
        img_array = np.array(img)
        
        print(f"📊 {filename}:")
        print(f"  - Size: {img.size[0]} x {img.size[1]}")
        print(f"  - Mode: {img.mode}")
        print(f"  - File size: {os.path.getsize(filename):,} bytes")
        
        # Check if image is mostly uniform (indicating empty/flat data)
        if len(img_array.shape) >= 3:
            # RGB/RGBA image
            gray = np.mean(img_array[:,:,:3], axis=2)
        else:
            # Grayscale
            gray = img_array
            
        variance = np.var(gray)
        print(f"  - Variance (content richness): {variance:.2f}")
        
        if variance < 100:
            print("  ⚠️  LOW VARIANCE - May indicate flat/empty data")
        else:
            print("  ✅ GOOD VARIANCE - Contains meaningful visual data")
            
        # Check for common flat line patterns
        height, width = gray.shape
        horizontal_lines = 0
        for i in range(height):
            row_var = np.var(gray[i, :])
            if row_var < 1:  # Very flat row
                horizontal_lines += 1
                
        flat_ratio = horizontal_lines / height
        print(f"  - Flat horizontal lines: {flat_ratio:.1%}")
        
        if flat_ratio > 0.8:
            print("  ❌ HIGH FLAT RATIO - Likely contains flat line artifacts")
        else:
            print("  ✅ GOOD VARIATION - Not dominated by flat lines")
            
    except Exception as e:
        print(f"❌ Error analyzing {filename}: {e}")

if __name__ == "__main__":
    print("Analyzing PNG file contents...")
    print("=" * 50)
    
    files_to_check = [
        "experiment2_dispersion_bandwidth.png",
        "experiment4_parameter_sensitivity.png"
    ]
    
    for filename in files_to_check:
        analyze_png(filename)
        print() 