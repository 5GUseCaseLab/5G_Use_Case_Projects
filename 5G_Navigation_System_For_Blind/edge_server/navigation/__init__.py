"""
Navigation Module Package for 5G Assistive Wearable
===================================================
Provides end-to-end walking navigation, routing, landmark snapping, and voice:
- OSRM walking router and waypoint tracker.
- Campus landmark fuzzy snapping (IIT Guwahati).
- Speech-to-Text via Multilingual Whisper AI ('small') & Google Speech fallback.
- Text-to-Speech via offline Windows SAPI audio synthesis.
- Initial and fallback GPS coordinate location service.
"""

from .osrm_routing import get_detailed_route, geocode_location, haversine_distance_km, IITG_CAMPUS_COORDINATES
from .campus_snapping import snap_to_campus_landmark, get_search_terms_for_landmark, CAMPUS_LANDMARKS
from .speech_to_text import get_voice_command, transcribe_audio_np, decode_audio_to_np, get_whisper_model, clean_destination_query
from .text_to_speech import speak_text
from .location_service import get_live_location
from .route_optimizer import route_cost, remaining_cost, is_clearly_better

__all__ = [
    'get_detailed_route',
    'geocode_location',
    'haversine_distance_km',
    'IITG_CAMPUS_COORDINATES',
    'snap_to_campus_landmark',
    'get_search_terms_for_landmark',
    'CAMPUS_LANDMARKS',
    'get_voice_command',
    'transcribe_audio_np',
    'decode_audio_to_np',
    'get_whisper_model',
    'clean_destination_query',
    'speak_text',
    'get_live_location',
    'route_cost',
    'remaining_cost',
    'is_clearly_better'
]
