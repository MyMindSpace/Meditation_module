"""
Bridge script to update fused_decision.json with meditation selector results.
This ensures the meditation script generator gets the correct meditation type.
"""
import json
import os

def update_fused_decision_with_recommendation():
    """Update fused_decision.json with the top recommendation from meditation selector."""
    
    # Load meditation recommendations
    try:
        with open('meditation_recommendations_output.json', 'r') as f:
            recommendations = json.load(f)
        print("✓ Loaded meditation recommendations")
    except FileNotFoundError:
        print("❌ meditation_recommendations_output.json not found")
        return False
    except Exception as e:
        print(f"❌ Error loading recommendations: {e}")
        return False
    
    # Load existing fused decision
    try:
        with open('preprocess_output/fused_decision.json', 'r') as f:
            fused_decision = json.load(f)
        print("✓ Loaded existing fused decision")
    except FileNotFoundError:
        print("❌ preprocess_output/fused_decision.json not found")
        return False
    except Exception as e:
        print(f"❌ Error loading fused decision: {e}")
        return False
    
    # Extract top recommendation
    if 'recommendations' in recommendations and len(recommendations['recommendations']) > 0:
        top_recommendation = recommendations['recommendations'][0]
        meditation_type = top_recommendation['meditation_type']
        confidence = top_recommendation['confidence']
        
        print(f"✓ Top recommendation: {meditation_type} (confidence: {confidence})")
        
        # Update fused decision with the new meditation type
        if 'inputs' not in fused_decision:
            fused_decision['inputs'] = {}
        if 'msm' not in fused_decision['inputs']:
            fused_decision['inputs']['msm'] = {}
        
        fused_decision['inputs']['msm']['label'] = meditation_type
        fused_decision['inputs']['msm']['confidence'] = confidence
        fused_decision['inputs']['msm']['source'] = 'meditation_selector'
        
        # Save updated fused decision
        try:
            with open('preprocess_output/fused_decision.json', 'w') as f:
                json.dump(fused_decision, f, indent=2, ensure_ascii=False)
            print(f"✓ Updated fused_decision.json with meditation type: {meditation_type}")
            return True
        except Exception as e:
            print(f"❌ Error saving updated fused decision: {e}")
            return False
    else:
        print("❌ No recommendations found in meditation recommendations")
        return False

if __name__ == "__main__":
    success = update_fused_decision_with_recommendation()
    if success:
        print("✅ Bridge script completed successfully!")
    else:
        print("❌ Bridge script failed!")
