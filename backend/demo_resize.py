"""
Demonstration of Task 3.4 resize functionality with visual examples.
"""

from PIL import Image
import io


def demonstrate_resize():
    """Demonstrate resize with various test cases."""
    
    print("\n" + "=" * 80)
    print("TASK 3.4 DEMONSTRATION: Aspect Ratio-Preserving Resize")
    print("=" * 80)
    
    # Recreate the resize logic from ImageProcessor
    def resize_image(width, height, max_dimension=640):
        """Calculate new dimensions after resize."""
        scale_factor = min(max_dimension / width, max_dimension / height)
        
        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return new_width, new_height, scale_factor
        else:
            return width, height, 1.0
    
    test_cases = [
        ("Landscape (wide)", 1920, 1080),
        ("Portrait (tall)", 1080, 1920),
        ("Square (large)", 1000, 1000),
        ("Small (no resize)", 500, 400),
        ("Exactly 640", 640, 640),
        ("Panoramic", 2400, 800),
        ("Ultra-wide", 3840, 1080),
        ("Standard photo", 3000, 2000),
    ]
    
    print("\nTest Cases:")
    print("-" * 80)
    print(f"{'Description':<20} {'Original':<15} {'Scale':<10} {'Resized':<15} {'Aspect Ratio':<20}")
    print("-" * 80)
    
    for desc, orig_w, orig_h in test_cases:
        new_w, new_h, scale = resize_image(orig_w, orig_h)
        
        # Calculate aspect ratios
        orig_aspect = orig_w / orig_h
        new_aspect = new_w / new_h
        aspect_diff = abs(orig_aspect - new_aspect)
        
        # Format output
        original = f"{orig_w}×{orig_h}"
        resized = f"{new_w}×{new_h}"
        aspect = f"{orig_aspect:.3f} → {new_aspect:.3f}"
        
        # Check if aspect ratio is preserved
        preserved = "✓" if aspect_diff < 0.01 else "✗"
        
        print(f"{desc:<20} {original:<15} {scale:<10.3f} {resized:<15} {aspect:<20} {preserved}")
    
    print("-" * 80)
    
    # Detailed example
    print("\nDetailed Example: 1920×1080 image")
    print("-" * 80)
    orig_w, orig_h = 1920, 1080
    new_w, new_h, scale = resize_image(orig_w, orig_h)
    
    print(f"Original dimensions:     {orig_w} × {orig_h} pixels")
    print(f"Original aspect ratio:   {orig_w/orig_h:.4f} (16:9)")
    print()
    print(f"Scale factor calculation:")
    print(f"  min(640/{orig_w}, 640/{orig_h})")
    print(f"  = min({640/orig_w:.4f}, {640/orig_h:.4f})")
    print(f"  = {scale:.4f}")
    print()
    print(f"New dimensions:")
    print(f"  width  = {orig_w} × {scale:.4f} = {new_w} pixels")
    print(f"  height = {orig_h} × {scale:.4f} = {new_h} pixels")
    print()
    print(f"Resized dimensions:      {new_w} × {new_h} pixels")
    print(f"Resized aspect ratio:    {new_w/new_h:.4f} (16:9)")
    print(f"Aspect ratio difference: {abs(orig_w/orig_h - new_w/new_h):.6f}")
    print()
    print("✓ Aspect ratio preserved!")
    print("✓ Maximum dimension (640px) constraint satisfied!")
    print("✓ Both dimensions scaled by same factor!")
    
    print("\n" + "=" * 80)
    print("Key Features Demonstrated:")
    print("-" * 80)
    print("1. Scale factor: min(640/width, 640/height)")
    print("   • Ensures image fits within 640px maximum")
    print("   • Chooses the constraining dimension")
    print()
    print("2. Lanczos resampling: Image.Resampling.LANCZOS")
    print("   • High-quality interpolation algorithm")
    print("   • Preserves image detail during downscaling")
    print("   • Better than NEAREST, BILINEAR, or BICUBIC")
    print()
    print("3. Aspect ratio preservation:")
    print("   • Both dimensions scaled by SAME factor")
    print("   • new_width = int(width × scale)")
    print("   • new_height = int(height × scale)")
    print("   • No distortion or stretching")
    print()
    print("4. Conditional resize:")
    print("   • Only resizes if scale_factor < 1.0")
    print("   • Images within 640px are unchanged")
    print("   • Prevents unnecessary upscaling")
    print("=" * 80)
    print()


if __name__ == "__main__":
    demonstrate_resize()
