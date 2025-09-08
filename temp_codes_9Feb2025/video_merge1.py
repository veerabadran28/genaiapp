#!/usr/bin/env python3
"""
FFmpeg-based Video Merger - More reliable alternative
Requires: pip install ffmpeg-python
"""

import ffmpeg
import os
import sys
import tempfile
from pathlib import Path
import argparse

class FFmpegVideoMerger:
    """A reliable video merger using ffmpeg-python."""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v']
    
    def validate_file(self, filepath: str) -> bool:
        """Validate if the file exists and has supported format."""
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File '{filepath}' not found.")
            return False
        
        if path.suffix.lower() not in self.supported_formats:
            print(f"Error: Unsupported format '{path.suffix}'. Supported: {self.supported_formats}")
            return False
        
        return True
    
    def get_video_info(self, filepath: str) -> dict:
        """Get video information using ffprobe."""
        try:
            probe = ffmpeg.probe(filepath)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            info = {
                'duration': float(probe['format']['duration']),
                'size': probe['format']['size'],
                'width': video_stream['width'] if video_stream else 0,
                'height': video_stream['height'] if video_stream else 0,
                'has_audio': audio_stream is not None
            }
            return info
        except Exception as e:
            print(f"Error getting video info for {filepath}: {e}")
            return {}
    
    def merge_videos_simple(self, input_files: list, output_file: str) -> bool:
        """
        Simple concatenation using FFmpeg's concat filter.
        This is the most reliable method.
        """
        try:
            # Validate all input files
            for file in input_files:
                if not self.validate_file(file):
                    return False
            
            print(f"Merging {len(input_files)} videos...")
            
            # Show video information
            for i, file in enumerate(input_files):
                info = self.get_video_info(file)
                if info:
                    print(f"Video {i+1}: {Path(file).name}")
                    print(f"  Duration: {info['duration']:.2f}s, Resolution: {info['width']}x{info['height']}")
            
            # Create input streams
            inputs = []
            for file in input_files:
                inputs.append(ffmpeg.input(file))
            
            # Concatenate videos
            joined = ffmpeg.concat(*inputs, v=1, a=1).node
            output = ffmpeg.output(joined['v'], joined['a'], output_file)
            
            # Run the ffmpeg command
            print(f"Writing merged video to '{output_file}'...")
            ffmpeg.run(output, overwrite_output=True, quiet=True)
            
            print(f"✅ Successfully merged videos into '{output_file}'")
            return True
            
        except ffmpeg.Error as e:
            print(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except Exception as e:
            print(f"❌ Error during merging: {str(e)}")
            return False
    
    def merge_videos_with_file_list(self, input_files: list, output_file: str) -> bool:
        """
        Alternative method using FFmpeg's concat demuxer (file list method).
        Good for when videos have different formats/codecs.
        """
        try:
            # Validate all input files
            for file in input_files:
                if not self.validate_file(file):
                    return False
            
            # Create temporary file list
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for file in input_files:
                    # Convert to absolute path and escape for FFmpeg
                    abs_path = os.path.abspath(file)
                    f.write(f"file '{abs_path}'\n")
                temp_file = f.name
            
            try:
                print(f"Merging {len(input_files)} videos using file list method...")
                
                # Use concat demuxer
                (
                    ffmpeg
                    .input(temp_file, format='concat', safe=0)
                    .output(output_file, c='copy')
                    .run(overwrite_output=True, quiet=True)
                )
                
                print(f"✅ Successfully merged videos into '{output_file}'")
                return True
                
            finally:
                # Clean up temp file
                os.unlink(temp_file)
                
        except ffmpeg.Error as e:
            print(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except Exception as e:
            print(f"❌ Error during merging: {str(e)}")
            return False
    
    def merge_videos_re_encode(self, input_files: list, output_file: str) -> bool:
        """
        Merge videos with re-encoding for maximum compatibility.
        Use this if other methods fail due to codec differences.
        """
        try:
            # Validate all input files
            for file in input_files:
                if not self.validate_file(file):
                    return False
            
            print(f"Merging {len(input_files)} videos with re-encoding...")
            print("Note: This method is slower but ensures compatibility.")
            
            # Create input streams
            inputs = []
            for file in input_files:
                inputs.append(ffmpeg.input(file))
            
            # Concatenate and re-encode
            joined = ffmpeg.concat(*inputs, v=1, a=1).node
            output = ffmpeg.output(
                joined['v'], joined['a'], output_file,
                vcodec='libx264',
                acodec='aac',
                preset='medium',
                crf=23
            )
            
            print(f"Writing merged video to '{output_file}'...")
            ffmpeg.run(output, overwrite_output=True)
            
            print(f"✅ Successfully merged videos into '{output_file}'")
            return True
            
        except ffmpeg.Error as e:
            print(f"❌ FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return False
        except Exception as e:
            print(f"❌ Error during merging: {str(e)}")
            return False

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Merge video files using FFmpeg")
    parser.add_argument("input_files", nargs='+', help="Input video files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("-m", "--method", 
                       choices=['simple', 'filelist', 'reencode'], 
                       default='simple', 
                       help="Merge method (default: simple)")
    
    args = parser.parse_args()
    
    # Create merger instance
    merger = FFmpegVideoMerger()
    
    # Try the specified method
    success = False
    if args.method == 'simple':
        success = merger.merge_videos_simple(args.input_files, args.output)
    elif args.method == 'filelist':
        success = merger.merge_videos_with_file_list(args.input_files, args.output)
    elif args.method == 'reencode':
        success = merger.merge_videos_re_encode(args.input_files, args.output)
    
    # If the chosen method fails, try alternatives
    if not success:
        print("\nFirst method failed. Trying alternative approaches...")
        
        if args.method != 'filelist':
            print("Trying file list method...")
            success = merger.merge_videos_with_file_list(args.input_files, args.output)
        
        if not success and args.method != 'reencode':
            print("Trying re-encode method...")
            success = merger.merge_videos_re_encode(args.input_files, args.output)
    
    sys.exit(0 if success else 1)

# Simple usage example
def simple_example():
    """Simple example for merging two videos."""
    merger = FFmpegVideoMerger()
    
    # Example usage - replace with your actual file paths
    input_files = ["video1.mp4", "video2.mp4"]
    output_file = "merged_output.mp4"
    
    # Try simple method first
    if not merger.merge_videos_simple(input_files, output_file):
        print("Simple method failed, trying file list method...")
        merger.merge_videos_with_file_list(input_files, output_file)

if __name__ == "__main__":
    # Uncomment for simple usage without command line args
    # simple_example()
    
    # Command line interface
    main()
