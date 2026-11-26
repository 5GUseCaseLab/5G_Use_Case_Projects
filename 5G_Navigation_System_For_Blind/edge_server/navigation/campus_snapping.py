"""
Campus Snapping Module for 5G Assistive Wearable
================================================
Performs intelligent fuzzy landmark matching against the IIT Guwahati 
campus landmarks registry (campus_landmarks.json). Resolves misheard 
or abbreviated spoken destinations to canonical campus locations.
"""

import os
import json
import re
import sys
from difflib import SequenceMatcher

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def _load_campus_landmarks():
    """Loads all canonical landmarks and search aliases from campus_landmarks.json."""
    landmarks_file = os.path.join(os.path.dirname(__file__), "campus_landmarks.json")
    if os.path.exists(landmarks_file):
        try:
            with open(landmarks_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Notice: Could not parse campus_landmarks.json: {e}")
    return []

CAMPUS_LANDMARKS = _load_campus_landmarks()

def _clean_str(text):
    """Normalize text: lowercased, letters & digits only."""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def _token_similarity(s1, s2):
    """
    Computes bidirectional token and sequence similarity (0.0 to 1.0).
    Captures stem matches, phonetic mishearings, and phrase overlaps.
    """
    clean1 = _clean_str(s1)
    clean2 = _clean_str(s2)
    
    if not clean1 or not clean2:
        return 0.0
        
    # Exact match
    if clean1 == clean2:
        return 1.0
        
    # Substring match (e.g. 'library' in 'central library')
    if clean1 in clean2 or clean2 in clean1:
        return 0.90
        
    # Sequence ratio
    seq_ratio = SequenceMatcher(None, clean1, clean2).ratio()
    
    # Token-level overlap (e.g., 'student activities until' vs 'student activity centre')
    tokens1 = set(clean1.split())
    tokens2 = set(clean2.split())
    
    # Measure stem overlap (words sharing >= 4 characters, e.g., 'activit')
    stem_matches = 0
    for t1 in tokens1:
        for t2 in tokens2:
            if len(t1) >= 4 and len(t2) >= 4 and (t1.startswith(t2[:4]) or t2.startswith(t1[:4])):
                stem_matches += 1
                break
                
    token_score = (2.0 * stem_matches) / (len(tokens1) + len(tokens2)) if (tokens1 and tokens2) else 0.0
    
    # Return weighted blend
    return max(seq_ratio, 0.4 * seq_ratio + 0.6 * token_score)

def snap_to_campus_landmark(spoken_text, threshold=0.72):
    """
    Matches natural voice input against verified IIT Guwahati landmarks.
    Returns: (snapped_destination, confidence_percentage, was_corrected_flag)
    """
    if not spoken_text:
        return spoken_text, 0, False
        
    cleaned_input = _clean_str(spoken_text)
    
    # Direct fast-path for common spoken short forms & phonetic substitutions
    special_cases = {
        # SAC / Student Activity Centre phonetic variants & mishearings
        "sac": "New SAC",
        "new sac": "New SAC",
        "newsac": "New SAC",
        "old sac": "Old SAC",
        "oldsac": "Old SAC",
        "new set": "New SAC",
        "old set": "Old SAC",
        "insat": "New SAC",
        "in sat": "New SAC",
        "new sat": "New SAC",
        "newsat": "New SAC",
        "use up": "New SAC",
        "use sac": "New SAC",
        "new sack": "New SAC",
        "sack": "New SAC",
        "nu sac": "New SAC",
        "newsec": "New SAC",
        "new sec": "New SAC",
        "student activities until": "New SAC",
        "student activity centre": "New SAC",
        "student activity center": "New SAC",
        "student activity": "New SAC",
        "student activities": "New SAC",
        "sac building": "New SAC",
        
        # Academic & Lecture Halls
        "lhc": "Lecture Hall Complex",
        "lecture hall": "Lecture Hall Complex",
        "lecture halls": "Lecture Hall Complex",
        "lecture hall complex": "Lecture Hall Complex",
        "lecture complex": "Lecture Hall Complex",
        
        # Library
        "library": "Central Library",
        "central library": "Central Library",
        "central lib": "Central Library",
        "cl": "Central Library",
        
        # Medical & Health
        "hospital": "Health Centre",
        "health centre": "Health Centre",
        "health center": "Health Centre",
        "dispensary": "Health Centre",
        "clinic": "Health Centre",
        
        # Food & Canteens
        "fc": "Food Court",
        "food court": "Food Court",
        "foodcourt": "Food Court",
        "canteen": "Food Court",
        "khoka": "Khokha Market",
        "khokha": "Khokha Market",
        "khoka market": "Khokha Market",
        "khokha market": "Khokha Market",
        
        # Gates
        "main gate": "Main Gate",
        "front gate": "Main Gate",
        "campus gate": "Main Gate",
        
        # Cores & Departments
        "dod": "Department of Design",
        "department of design": "Department of Design",
        "design department": "Department of Design",
        "core1": "Core 1",
        "core 1": "Core 1",
        "core one": "Core 1",
        "core2": "Core 2",
        "core 2": "Core 2",
        "core two": "Core 2",
        "core to": "Core 2",
        "core too": "Core 2",
        "core3": "Core 3",
        "core 3": "Core 3",
        "core three": "Core 3",
        "core4": "Core 4",
        "core 4": "Core 4",
        "core four": "Core 4",
        "core for": "Core 4",
        
        # Hostels
        "brahma": "Brahmaputra Hostel",
        "brahmaputra": "Brahmaputra Hostel",
        "brahmaputr": "Brahmaputra Hostel",
        "brahmaputra hostel": "Brahmaputra Hostel",
        "dihing": "Dihing Hostel",
        "dihing hostel": "Dihing Hostel",
        "kameng": "Kameng Hostel",
        "kameng hostel": "Kameng Hostel",
        "subansiri": "Subansiri Hostel",
        "subansiri hostel": "Subansiri Hostel",
        "manas": "Manas Hostel",
        "manas hostel": "Manas Hostel",
        "umiam": "Umiam Hostel",
        "umiam hostel": "Umiam Hostel",
        "barak": "Barak Hostel",
        "barak hostel": "Barak Hostel",
        "kapili": "Kapili Hostel",
        "kapili hostel": "Kapili Hostel",
        "siang": "Siang Hostel",
        "siang hostel": "Siang Hostel",
        "lohit": "Lohit Hostel",
        "lohit hostel": "Lohit Hostel",
        "dibang": "Dibang Hostel",
        "dibang hostel": "Dibang Hostel",
        "dhansiri": "Dhansiri Hostel",
        "dhansiri hostel": "Dhansiri Hostel",
        "disang": "Disang Hostel",
        "disang hostel": "Disang Hostel",
        "msh": "Married Scholars Hostel",
        "married scholars hostel": "Married Scholars Hostel"
    }
    
    if cleaned_input in special_cases:
        return special_cases[cleaned_input], 98, True
        
    best_candidate = None
    best_score = 0.0
    
    for item in CAMPUS_LANDMARKS:
        canonical = item["canonical"]
        # Compare against canonical name
        score = _token_similarity(cleaned_input, canonical)
        
        # Compare against aliases
        for alias in item.get("aliases", []):
            alias_score = _token_similarity(cleaned_input, alias)
            if alias_score > score:
                score = alias_score
                
        if score > best_score:
            best_score = score
            best_candidate = canonical
            
    confidence = int(best_score * 100)
    # Stricter threshold for short words (< 7 chars) to prevent anagram collisions
    effective_threshold = 0.80 if len(cleaned_input) <= 6 else threshold
    
    if best_score >= effective_threshold and best_candidate:
        was_corrected = (cleaned_input != _clean_str(best_candidate))
        return best_candidate, confidence, was_corrected
        
    return spoken_text, confidence, False

def get_search_terms_for_landmark(location_name):
    """
    Returns an exhaustive list of search query strings (canonical + all aliases + spelling variants)
    for geocoding against OpenStreetMap.
    """
    terms = [location_name]
    norm_loc = _clean_str(location_name)
    
    # 1. Match against CAMPUS_LANDMARKS registry
    matched_entry = None
    for entry in CAMPUS_LANDMARKS:
        canon_clean = _clean_str(entry["canonical"])
        if norm_loc == canon_clean or any(norm_loc == _clean_str(a) for a in entry.get("aliases", [])):
            matched_entry = entry
            break
            
    if matched_entry:
        if matched_entry["canonical"] not in terms:
            terms.insert(0, matched_entry["canonical"])
        for alias in matched_entry.get("aliases", []):
            title_alias = alias.title()
            if title_alias not in terms and len(title_alias) >= 3:
                terms.append(title_alias)
                
    # 2. Add automatic Centre <-> Center variants
    extra_variants = []
    for t in list(terms):
        if "Centre" in t:
            extra_variants.append(t.replace("Centre", "Center"))
        elif "Center" in t:
            extra_variants.append(t.replace("Center", "Centre"))
            
        if "Students" in t:
            extra_variants.append(t.replace("Students", "Student"))
        elif "Student " in t:
            extra_variants.append(t.replace("Student ", "Students "))
            
        if "Hostel" in t:
            extra_variants.append(t.replace(" Hostel", "").strip())
        elif matched_entry and "hostel" in matched_entry["canonical"].lower():
            extra_variants.append(f"{t} Hostel")
            
        if "Department of " in t:
            dept_name = t.replace("Department of ", "").strip()
            extra_variants.append(f"{dept_name} Department")
            extra_variants.append(dept_name)
            
    for ev in extra_variants:
        if ev not in terms and len(ev) >= 3:
            terms.append(ev)
            
    return terms

if __name__ == "__main__":
    test_phrases = [
        "Student activities until",
        "New set",
        "New SAC",
        "Take me to Brahmaputr",
        "Where is LHC",
        "Go to Core to",
        "Central lib",
        "Food court",
        "Khokha",
        "hospital"
    ]
    
    print("--- Testing Campus Fuzzy Snapping Engine ---")
    for phrase in test_phrases:
        snapped, conf, corrected = snap_to_campus_landmark(phrase)
        tag = "⚡ [SNAPPED]" if corrected else "✓ [EXACT]"
        print(f"{tag} '{phrase}' -> '{snapped}' (Confidence: {conf}%)")
