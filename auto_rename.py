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

def extract_video_frame_grids(video_path, temp_img_path_1, temp_img_path_2, temp_img_path_3, temp_img_path_4, temp_fullres_path):
    """Extracts 16 sequential frames into four 2x2 grids, plus a full-HD frame at the 50% mark."""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return False

        # Four grids of 4 frames each, evenly spread across the whole clip:
        # Grid 1 covers 0-25%, Grid 2 covers 25-50%, Grid 3 covers 50-75%, Grid 4 covers 75-100%
        sample_points = [0.03, 0.10, 0.16, 0.23, 0.29, 0.36, 0.43, 0.50, 0.56, 0.63, 0.70, 0.76, 0.83, 0.90, 0.96, 0.98]
        frames = []
        fullres_saved = False

        for ratio in sample_points:
            target_frame = int(total_frames * ratio)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret:
                # Save a full-HD frame at the 50% mark so the model can see exact contents/detail
                if ratio == 0.50:
                    height, width = frame.shape[:2]
                    max_w, max_h = 1920, 1080
                    if width > max_w or height > max_h:
                        scale = min(max_w / width, max_h / height)
                        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                    fullres_saved = cv2.imwrite(temp_fullres_path, frame)
                # Downscale each frame to 320x180 so the combined 640x360 grid stays fast
                resized = cv2.resize(frame, (320, 180))
                frames.append(resized)

        if len(frames) == 16:
            def make_grid(f):
                return cv2.vconcat([cv2.hconcat([f[0], f[1]]), cv2.hconcat([f[2], f[3]])])

            cv2.imwrite(temp_img_path_1, make_grid(frames[:4]))
            cv2.imwrite(temp_img_path_2, make_grid(frames[4:8]))
            cv2.imwrite(temp_img_path_3, make_grid(frames[8:12]))
            cv2.imwrite(temp_img_path_4, make_grid(frames[12:]))
            return fullres_saved

    except Exception as e:
        print(f"   ⚠️ OpenCV frame grid extraction error: {e}")
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
    temp_frame_path_1 = None
    temp_frame_path_2 = None
    temp_frame_path_3 = None
    temp_frame_path_4 = None
    temp_fullres_path = None

    if is_video:
        temp_dir = tempfile.gettempdir()
        temp_frame_path_1 = os.path.join(temp_dir, "ollama_temp_frame_1.jpg")
        temp_frame_path_2 = os.path.join(temp_dir, "ollama_temp_frame_2.jpg")
        temp_frame_path_3 = os.path.join(temp_dir, "ollama_temp_frame_3.jpg")
        temp_frame_path_4 = os.path.join(temp_dir, "ollama_temp_frame_4.jpg")
        temp_fullres_path = os.path.join(temp_dir, "ollama_temp_frame_fullhd.jpg")
        print("   -> [STEP 1/3] Video detected. Extracting 16-frame action sequence grids + full-HD mid frame...")
        if not extract_video_frame_grids(file_path, temp_frame_path_1, temp_frame_path_2, temp_frame_path_3, temp_frame_path_4, temp_fullres_path):
            return None
        images_to_send = [temp_frame_path_1, temp_frame_path_2, temp_frame_path_3, temp_frame_path_4, temp_fullres_path]

        prompt = (
            "You are a stock footage labeler. You are given FIVE images from the same video clip.\n"
            "Images 1-4 are 2x2 grids of frames covering the clip in chronological order:\n"
            "Image 1 covers 0-25% of the clip.\n"
            "Image 2 covers 25-50% of the clip.\n"
            "Image 3 covers 50-75% of the clip.\n"
            "Image 4 covers 75-100% of the clip.\n"
            "Chronological order within each grid: top-left → top-right → bottom-left → bottom-right.\n"
            "Image 5: a single FULL-RESOLUTION frame captured exactly at the 50% mark. Use it to identify "
            "the exact contents, subjects, and fine details that the low-res grids may miss.\n\n"

            "IGNORE: Any facecam, webcam overlay, picture-in-picture window, or person shown in a small "
            "corner box (circle or square). These are not the subject.\n\n"

            "DESCRIBE: The primary action, motion, or subject visible across all five images. "
            "If the action changes significantly between the opening, middle, and ending, prioritize the dominant "
            "or most visually distinct one. Include the specific objects and scene details visible in the "
            "full-resolution frame.\n\n"

            "CRITICAL — NAME THE EXACT GESTURE AND ITS MEANING: When a person is gesturing or "
            "signaling to the camera, do NOT say just 'gesturing_with_hands'. Name the precise gesture and what it "
            "communicates. Look at the sequence of frames closely to decide. Examples of specific gestures: "
            "'no' (waving hands side-to-side, wagging index finger, or shaking head), 'yes' (nodding or thumbs up), "
            "'stop', 'waving hello', 'pointing up', 'thumbs down', 'come here', 'talking on phone', "
            "'shushing' (finger to lips), 'crossed arms'. If the gesture is recognizable, its meaning MUST appear "
            "in the output (e.g. a 'no' gesture should produce something like 'man_shaking_head_no' or "
            "'woman_waving_hands_no').\n\n"

            "OUTPUT RULES:\n"
            "- Exactly 4 to 6 words\n"
            "- Lowercase, words separated by underscores\n"
            "- No punctuation, no filler words (a, the, is, are)\n"
            "- Start with an action verb or subject noun\n"
            "- Output the label only — no explanation, no extra text\n\n"

            "EXAMPLES OF GOOD OUTPUT:\n"
            "dog_running_on_beach\n"
            "drone_shot_city_skyline\n"
            "man_shaking_head_no\n"
            "woman_waving_hands_no\n"
            "man_thumbs_up_approval\n"
            "person_waving_hello_camera\n"
            "crowd_cheering_at_concert\n"
            "car_driving_mountain_road"
        )
    else:
        images_to_send = [file_path]
        prompt = (
            "You are a stock photo labeler. Analyze this image and generate a highly searchable filename.\n\n"

            "DESCRIBE: The primary subject, setting, and key identifying features such as object, location, action, or color. "
            "Prioritize high-value keywords for file indexing over generic terms.\n\n"

            "OUTPUT RULES:\n"
            "- Exactly 4 to 6 words\n"
            "- Lowercase, words separated by underscores\n"
            "- No punctuation, no filler words (a, the, is, are)\n"
            "- Start with the most specific subject noun\n"
            "- Output the label only — no explanation, no extra text\n\n"

            "EXAMPLES OF GOOD OUTPUT:\n"
            "golden_retriever_sunset_beach\n"
            "amazon_invoice_receipt_document\n"
            "red_vintage_convertible_sports_car\n"
            "snow_covered_mountain_peak\n"
            "fresh_fruit_wooden_market_stall"
        )

    try:
        print("   -> [STEP 2/3] Payload ready. Querying local Ollama API (waiting for response)...")

        response = client.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': images_to_send
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
        for path in [temp_frame_path_1, temp_frame_path_2, temp_frame_path_3, temp_frame_path_4, temp_fullres_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
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

    renamed_dir = os.path.join(folder_path, "renamed")
    os.makedirs(renamed_dir, exist_ok=True)

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
        unique_name = get_unique_filename(renamed_dir, clean_base, ext)
        
        if dry_run:
            print(f"   👉 [PLAN] '{original_name}' ➔ '{os.path.join('renamed', unique_name)}'\n")
            success_count += 1
        else:
            try:
                new_filepath = os.path.join(renamed_dir, unique_name)
                os.rename(filepath, new_filepath)
                print(f"   ✅ Moved & renamed to: '{os.path.join('renamed', unique_name)}'\n")
                success_count += 1
            except Exception as e:
                print(f"   ❌ Error renaming file: {e}\n")
        
        time.sleep(1.0)

    print("-" * 60)
    if dry_run:
        print(f"🎉 Dry run complete. {success_count}/{len(queue)} files mapped.")
        run_real = input("❓ Proceed with actual renaming? [y/n]: ").lower()
        if run_real == 'y':
            main_run_no_dry(folder_path, renamed_dir, queue)
    else:
        print(f"🎉 Process completed. Renamed {success_count}/{len(queue)} files.")

def main_run_no_dry(folder_path, renamed_dir, queue):
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
        unique_name = get_unique_filename(renamed_dir, clean_base, ext)
        
        try:
            new_filepath = os.path.join(renamed_dir, unique_name)
            os.rename(current_filepath, new_filepath)
            print(f"   ✅ Moved & renamed: '{original_name}' ➔ '{os.path.join('renamed', unique_name)}'\n")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
            
        time.sleep(1.0)
            
    print("-" * 60)
    print(f"🎉 Done! Successfully renamed {success_count} files.")

if __name__ == "__main__":
    main()