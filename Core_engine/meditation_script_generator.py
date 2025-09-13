"""
Meditation Script Generator for TTS Module

QUICK SETUP - INSERT YOUR API KEY BELOW:
============================================
1. For Gemini API: Get key from https://makersuite.google.com/app/apikey
   Replace None with your actual key: GEMINI_API_KEY = "your_key_here"

2. Run the script: python meditation_script_generator.py
============================================
"""

import json
import pandas as pd
import os

# =============================================================================
# API KEY PLACEHOLDER - Insert your Gemini API key here
# =============================================================================
GEMINI_API_KEY = "AIzaSyBkhEgCoFQU8IAeIhHanSff76tdnGXJwu4"  # Replace with: "your_gemini_api_key_here"
# =============================================================================

try:
    import google.generativeai as genai
    HAVE_GEMINI = True
except ImportError:
    HAVE_GEMINI = False
    print("Warning: google.generativeai not installed. Install with: pip install google-generativeai")

def main():
    """Main function to generate meditation script."""
    print("MEDITATION SCRIPT GENERATOR")
    print("="*50)
    
    # Load fused decision JSON
    print("Loading fused_decision.json...")
    try:
        with open('preprocess_output/fused_decision.json', 'r') as f:
            fused_decision = json.load(f)
        print("[OK] Fused Decision (JSON) loaded successfully")
    except FileNotFoundError:
        print("[ERROR] preprocess_output/fused_decision.json not found.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading fused_decision.json: {e}")
        return
    
    # Load meditation CSV
    print("Loading meditation.csv...")
    try:
        meditation_df = pd.read_csv('Core_engine/meditation.csv')
        print("[OK] Meditation Data (CSV) loaded successfully")
        print(f"Available meditations: {meditation_df['Name'].head().tolist()}")
    except FileNotFoundError:
        print("[ERROR] Core_engine/meditation.csv not found.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading meditation.csv: {e}")
        return
    
    # Extract meditation type
    print("\n" + "="*50)
    print("EXTRACTING MEDITATION TYPE")
    print("="*50)
    
    meditation_type = None
    if 'inputs' in fused_decision and 'msm' in fused_decision['inputs']:
        msm_data = fused_decision['inputs']['msm']
        if 'label' in msm_data:
            meditation_type = msm_data['label']
    
    print(f"Extracted Meditation Type: {meditation_type}")
    
    if meditation_type is None:
        print("✗ Cannot proceed without meditation type")
        return
    
    # Find meditation instruction
    print("\n" + "="*50)
    print("FINDING MEDITATION INSTRUCTION")
    print("="*50)
    
    try:
        matching_rows = meditation_df[meditation_df['Name'] == meditation_type]
        if matching_rows.empty:
            print(f"[ERROR] No meditation found with name: {meditation_type}")
            print(f"Available meditations: {meditation_df['Name'].tolist()[:5]}...")
            return
        meditation_instruction = matching_rows['Instructions'].iloc[0]
        print(f"Meditation Instructions: {meditation_instruction}")
    except Exception as e:
        print(f"[ERROR] Error finding meditation instruction: {e}")
        return
    
    # Generate guided meditation script
    print("\n" + "="*50)
    print("GENERATING MEDITATION SCRIPT")
    print("="*50)
    
    # Check if Gemini API key is provided
    if GEMINI_API_KEY is None:
        print("[ERROR] No Gemini API key provided. Set GEMINI_API_KEY in the code.")
        print("Generating fallback template script...")
        generated_script = generate_fallback_script(meditation_type, meditation_instruction)
    elif not HAVE_GEMINI:
        print("[ERROR] Gemini not available. Install with: pip install google-generativeai")
        print("Generating fallback template script...")
        generated_script = generate_fallback_script(meditation_type, meditation_instruction)
    else:
        try:
            # Configure the Gemini API
            genai.configure(api_key=GEMINI_API_KEY)
            
            # Create the prompt for the LLM
            prompt = f"""Generate a 3-4 minute guided meditation script.
Meditation Type: {meditation_type}
Instructions: {meditation_instruction}

Please provide a script that is:
- Easy to follow and suitable for a 3-4 minute duration
- Written in a calm, soothing tone suitable for text-to-speech
- Includes natural pauses marked with [pause] for TTS timing
- Has clear guidance for breathing and body awareness
- Ends with a gentle return to awareness

Format as spoken text that will be read aloud by a TTS system."""
            
            print("Calling Gemini API...")
            
            # Generate the guided meditation script using the LLM
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
            generated_script = response.text
            
            print("[OK] Script generated successfully with Gemini!")
            
        except Exception as e:
            print(f"[ERROR] Error with Gemini API: {e}")
            print("Falling back to template script...")
            generated_script = generate_fallback_script(meditation_type, meditation_instruction)
    
    # Display the generated script
    print("\n" + "="*50)
    print("GENERATED GUIDED MEDITATION SCRIPT")
    print("="*50)
    print(generated_script)
    print("="*50)
    
    # Save script to file
    print("\n" + "="*50)
    print("SAVING SCRIPT")
    print("="*50)
    
    try:
        import datetime
        script_data = {
            "meditation_type": meditation_type,
            "instructions": meditation_instruction,
            "script": generated_script,
            "generated_at": datetime.datetime.now().isoformat(),
            "duration_minutes": "3-4",
            "format": "TTS-ready"
        }
        os.makedirs('Core_engine', exist_ok=True)
        with open('Core_engine/generated_meditation_script.json', 'w', encoding='utf-8') as f:
            json.dump(script_data, f, indent=2, ensure_ascii=False)
        print("[OK] Script saved to: Core_engine/generated_meditation_script.json")
    except Exception as e:
        print(f"[ERROR] Error saving script: {e}")
    
    print("\n[OK] Meditation script generation completed!")

def generate_fallback_script(meditation_type, meditation_instruction):
    """Generate a fallback script when Gemini is not available."""
    script = f"""Welcome to your {meditation_type} practice. [pause]

Find a comfortable position, either sitting or lying down. [pause]

Close your eyes gently and take three deep, calming breaths. [pause]

{meditation_instruction} [pause]

Allow yourself to settle into this practice naturally. If your mind wanders, that's perfectly normal. [pause]

Simply notice any thoughts that arise and gently return your attention to the meditation. [pause]

Continue with this peaceful awareness for the next few minutes. [pause]

[Extended pause for practice - approximately 2-3 minutes]

Now, slowly begin to bring your attention back to your surroundings. [pause]

Take a deep, refreshing breath and when you're ready, gently open your eyes. [pause]

Your {meditation_type} practice is complete. Take a moment to notice how you feel. [pause]

Carry this sense of peace with you as you continue your day."""
    
    return script

if __name__ == '__main__':
    main()