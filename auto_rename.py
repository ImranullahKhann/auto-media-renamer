import os
import re
import sys
import time
import tempfile
import cv2
from ollama import Client

# Configuration
MODEL_NAME = "qwen3.5:4b"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

# Initialize Ollama client with a strict 45-second timeout
try:
    client = Client(timeout=45.0)
except Exception as e:
    print(f"❌ Failed to initialize Ollama client: {e}")
    sys.exit(1)

def check_ollama_status():
    """Verify Ollama is running and the model is available."""
    try:
        models_response = client.list()
        models = []
        if hasattr(models_response, 'models'):
            models = [m.model for m in models_response.models]
        elif isinstance(models_response, dict) and 'models' in models_response:
            models = [m.get('model', m.get('name', '')) for m in models_response['models']]
            
        has_model = any(MODEL_NAME in m or "qwen3.5" in m for m in models)
        if not has_model:
            print(f"⚠️ Warning: '{MODEL_NAME}' was not explicitly detected in your local models.")
            print("-" * 60)
    except Exception as e:
        print("❌ Error: Could not connect to local Ollama instance.")
        print("Make sure Ollama is actively running.")
        sys.exit(1)

def extract_video_frame_grid(video_path, temp_img_path):
    """Extracts 4 sequential frames and stitches them into a 2x2 grid to capture action/motion."""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return False
        
        # Sample 4 points across the video duration (20%, 40%, 60%, 80%)
        sample_points = [0.2, 0.4, 0.6, 0.8]
        frames = []
        
        for ratio in sample_points:
            target_frame = int(total_frames * ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret:
                # Downscale each frame to 320x180 so the combined 640x360 grid stays fast
                resized = cv2.resize(frame, (320, 180))
                frames.append(resized)
                
        if len(frames) == 4:
            # Create a 2x2 contact sheet
            top_row = cv2.hconcat([frames[0], frames[1]])
            bottom_row = cv2.hconcat([frames[2], frames[3]])
            grid = cv2.vconcat([top_row, bottom_row])
            
            cv2.imwrite(temp_img_path, grid)
            return True
            
    except Exception as e:
        print(f"   ⚠️ OpenCV Frame grid extraction error: {e}")
    finally:
        if cap is not None:
            cap.release()
            
    return False

def sanitize_filename(name):
    """Cleans the LLM response, stripping reasoning thoughts and safe-naming."""
    if not name or not isinstance(name, str):
        return "unnamed_media"
    
    # Strip Qwen's hidden reasoning block if present
    name = re.sub(r'<think>.*?</think>', '', name, flags=re.DOTALL)
    
    # Strip markdown styling or quotes
    name = re.sub(r'[`"\'*]', '', name)
    
    # Remove directory paths or extensions
    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    
    # Strip common introductory phrases LLMs include
    prefixes_to_remove = [
        "here is a filename:", "suggested filename:", "filename:", 
        "here is the filename:", "suggested name:", "name:"
    ]
    name_lower = name.lower()
    for prefix in prefixes_to_remove:
        if name_lower.startswith(prefix):
            name = name[len(prefix):].strip()
            name_lower = name.lower()
            
    # Clean non-alphanumeric characters
    name = name.replace('-', '_')
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_{2,}', '_', name)
    name = name.strip('_').lower()
    
    if len(name) > 50:
        name = name[:50].rstrip('_')
        
    return name if name else "unnamed_media"

def get_unique_filename(directory, base_name, extension):
    """Generates a non-conflicting filename."""
    counter = 1
    new_name = f"{base_name}{extension}"
    new_path = os.path.join(directory, new_name)
    while os.path.exists(new_path):
        new_name = f"{base_name}_{counter}{extension}"
        new_path = os.path.join(directory, new_name)
        counter += 1
    return new_name

def analyze_media(file_path, is_video=False):
    """Sends the media file to Ollama for analysis with strict reasoning suppression."""
    temp_frame_path = None
    
    if is_video:
        temp_dir = tempfile.gettempdir()
        temp_frame_path = os.path.join(temp_dir, "ollama_temp_frame.jpg")
        print("   -> [STEP 1/3] Video detected. Extracting 4-frame action sequence grid...")
        if not extract_video_frame_grid(file_path, temp_frame_path):
            return None
        image_to_send = temp_frame_path
        
        prompt = (
            "This image is a 2x2 grid showing sequential frames from a video ordered chronologically top-left, top-right, bottom-left, bottom-right. "
            "Analyze the movement across frames and generate a highly searchable, 4 to 6 word description. "
            "Prioritize the core subject action, motion, or gesture taking place (e.g., person_waving_hands_no, dog_running_park, man_shaking_head). "
            "Format strictly using lowercase letters and underscores instead of spaces. Do not write any intro, conversational filler, or punctuation."
        )
    else:
        image_to_send = file_path
        prompt = (
            "Analyze this image and generate a highly searchable, 4 to 6 word description focusing on the primary subject, setting, and key identifying features (e.g., specific object, location, action, or color). Prioritize high-value keywords for file indexing over generic terms."
            "Format your response strictly using lowercase letters and underscores instead of spaces."
            f"""Examples:
                - Input: Photo of a Golden Retriever on a beach at sunset -> golden_retriever_sunset_sandy_beach
                - Input: Invoice document from Amazon -> amazon_receipt_invoice_document
                - Input: Red vintage sports car -> red_vintage_convertible_sports_car
            Do not write any introduction, conversational filler, punctuation, or file extensions."""
        )

    try:
        print("   -> [STEP 2/3] Payload ready. Querying local Ollama API (waiting for response)...")
        
        response = client.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_to_send]
            }],
            think=False,
            options={
                'temperature': 0.1,
                'num_predict': 40
            }
        )
        print("   -> [STEP 3/3] Response successfully received.")
        
        description = ""
        msg = getattr(response, 'message', None)
        if msg:
            content = getattr(msg, 'content', '')
            thinking = getattr(msg, 'thinking', '')
            description = content.strip() if content else thinking.strip()
                
        return description if description else None
        
    except Exception as e:
        print(f"   ❌ [ERROR] Ollama request failed or timed out: {e}")
        return None
    finally:
        if temp_frame_path and os.path.exists(temp_frame_path):
            try:
                os.remove(temp_frame_path)
            except OSError:
                pass

def main():
    print("=" * 60)
    print("🤖 Local AI Media Renamer (Optimized for Qwen 3.5) 🤖")
    print("=" * 60)
    
    check_ollama_status()
    
    while True:
        folder_path = input("📂 Enter the absolute path to your folder: ").strip()
        folder_path = folder_path.strip('"\'')
        if os.path.isdir(folder_path):
            folder_path = os.path.abspath(folder_path)
            break
        print("❌ Invalid directory. Please try again.")

    all_files = os.listdir(folder_path)
    queue = []
    
    for filename in all_files:
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                queue.append((filename, ext, filepath, False))
            elif ext in VIDEO_EXTENSIONS:
                queue.append((filename, ext, filepath, True))

    if not queue:
        print("No matching stock photos or video files found in that folder.")
        return

    print(f"\n🔍 Found {len(queue)} supported media file(s).")
    dry_run_input = input("❓ Run a Preview DRY RUN first? [y/n]: ").lower()
    dry_run = dry_run_input != 'n'
    
    print("\n🚀 Starting..." if not dry_run else "\n🔍 Starting Dry Run Preview...")
    print("-" * 60)

    success_count = 0
    for idx, (original_name, ext, filepath, is_video) in enumerate(queue, start=1):
        media_type = "Video" if is_video else "Image"
        print(f"[{idx}/{len(queue)}] Processing {media_type}: '{original_name}'")
        
        raw_description = analyze_media(filepath, is_video=is_video)
        
        if not raw_description:
            print(f"   ❌ Skipped: Could not generate description.\n")
            continue
            
        clean_base = sanitize_filename(raw_description)
        unique_name = get_unique_filename(folder_path, clean_base, ext)
        
        if dry_run:
            print(f"   👉 [PLAN] '{original_name}' ➔ '{unique_name}'\n")
            success_count += 1
        else:
            try:
                new_filepath = os.path.join(folder_path, unique_name)
                os.rename(filepath, new_filepath)
                print(f"   ✅ Renamed to: '{unique_name}'\n")
                success_count += 1
            except Exception as e:
                print(f"   ❌ Error renaming file: {e}\n")
        
        time.sleep(1.0)

    print("-" * 60)
    if dry_run:
        print(f"🎉 Dry run complete. {success_count}/{len(queue)} files mapped.")
        run_real = input("❓ Proceed with actual renaming? [y/n]: ").lower()
        if run_real == 'y':
            main_run_no_dry(folder_path, queue)
    else:
        print(f"🎉 Process completed. Renamed {success_count}/{len(queue)} files.")

def main_run_no_dry(folder_path, queue):
    print("\n🚀 Starting active renaming process...")
    print("-" * 60)
    success_count = 0
    for idx, (original_name, ext, filepath, is_video) in enumerate(queue, start=1):
        current_filepath = os.path.join(folder_path, original_name)
        if not os.path.exists(current_filepath):
            continue
            
        print(f"[{idx}/{len(queue)}] Processing: '{original_name}'")
        raw_description = analyze_media(current_filepath, is_video=is_video)
        if not raw_description:
            print()
            continue
            
        clean_base = sanitize_filename(raw_description)
        unique_name = get_unique_filename(folder_path, clean_base, ext)
        
        try:
            new_filepath = os.path.join(folder_path, unique_name)
            os.rename(current_filepath, new_filepath)
            print(f"   ✅ Renamed: '{original_name}' ➔ '{unique_name}'\n")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            
        time.sleep(1.0)
            
    print("-" * 60)
    print(f"🎉 Done! Successfully renamed {success_count} files.")

if __name__ == "__main__":
    main()