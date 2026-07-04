from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
import math
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

app = FastAPI(title=\"Sign Language Recognition API\")
api_router = APIRouter(prefix=\"/api\")


class Landmark(BaseModel):
    x: float
    y: float
    z: float = 0.0


class RecognizeRequest(BaseModel):
    landmarks: List[Landmark] = Field(..., description=\"21 hand landmarks from MediaPipe\")
    handedness: str = Field(default=\"Right\")


class RecognizeResponse(BaseModel):
    label: str            # short label e.g. \"I LOVE YOU\"
    meaning: str          # full meaning / description
    confidence: float
    debug: dict


def _dist(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _classify(lm: List[Landmark], handedness: str) -> RecognizeResponse:
    \"\"\"
    Rule-based sign recognizer that maps 21 MediaPipe hand landmarks to common
    signs used by deaf/mute community: HELLO, I LOVE YOU, THANK YOU, GOOD,
    OK, PEACE, CALL ME, STOP, FIST, POINT, ROCK ON, THREE, plus a few
    ASL letters.
    \"\"\"
    if len(lm) != 21:
        raise HTTPException(status_code=400, detail=\"Need exactly 21 landmarks\")

    # Palm scale for normalization
    palm = _dist(lm[0], lm[9]) or 1e-6

    # Finger up detection: tip noticeably above PIP joint (smaller y)
    def up(tip, pip):
        return (lm[pip].y - lm[tip].y) > 0.025

    index_up = up(8, 6)
    middle_up = up(12, 10)
    ring_up = up(16, 14)
    pinky_up = up(20, 18)

    # Thumb extended laterally from index MCP
    thumb_lateral = abs(lm[4].x - lm[5].x) / palm
    thumb_extended = thumb_lateral > 0.55

    # Thumb up vertically (above wrist meaningfully)
    thumb_up_vertical = (lm[0].y - lm[4].y) / palm > 0.9 and not thumb_lateral > 0.8

    # Distances
    d_thumb_index = _dist(lm[4], lm[8]) / palm
    d_thumb_pinky = _dist(lm[4], lm[20]) / palm
    d_index_middle = _dist(lm[8], lm[12]) / palm

    # Curl metric — small = tightly curled fist
    avg_curl = (
        _dist(lm[8], lm[5]) + _dist(lm[12], lm[9]) +
        _dist(lm[16], lm[13]) + _dist(lm[20], lm[17])
    ) / 4 / palm

    fingers = (index_up, middle_up, ring_up, pinky_up)
    debug = {
        \"fingers_up\": list(fingers),
        \"thumb_extended\": thumb_extended,
        \"thumb_up_vertical\": thumb_up_vertical,
        \"d_thumb_index\": round(d_thumb_index, 3),
        \"d_index_middle\": round(d_index_middle, 3),
        \"d_thumb_pinky\": round(d_thumb_pinky, 3),
        \"avg_curl\": round(avg_curl, 3),
        \"thumb_lateral\": round(thumb_lateral, 3),
    }

    label = \"UNKNOWN\"
    meaning = \"Hold a clearer sign in front of the camera\"
    conf = 0.5

    # ---- Rule-based classification (priority order matters) ----

    # I LOVE YOU — thumb + index + pinky out, middle + ring down (most iconic)
    if thumb_extended and index_up and pinky_up and not middle_up and not ring_up:
        label, meaning, conf = \"I LOVE YOU\", \"I love you ❤\", 0.95

    # ROCK ON — index + pinky up, thumb tucked
    elif index_up and pinky_up and not middle_up and not ring_up and not thumb_extended:
        label, meaning, conf = \"ROCK ON\", \"Rock on / Awesome\", 0.85

    # CALL ME / Y — thumb + pinky out only
    elif thumb_extended and pinky_up and not index_up and not middle_up and not ring_up:
        label, meaning, conf = \"CALL ME\", \"Call me (Y sign)\", 0.9

    # THREE — thumb + index + middle up (check before PEACE since both have index+middle)
    elif thumb_extended and index_up and middle_up and not ring_up and not pinky_up:
        label, meaning, conf = \"THREE\", \"Number 3\", 0.85

    # PEACE / V — index + middle up (thumb tucked), ring + pinky down
    elif index_up and middle_up and not ring_up and not pinky_up and not thumb_extended:
        if d_index_middle > 0.35:
            label, meaning, conf = \"PEACE\", \"Peace / Victory (V)\", 0.9
        else:
            label, meaning, conf = \"TWO\", \"Number 2 / U\", 0.8

    # OK — thumb tip touches index tip, others up
    elif d_thumb_index < 0.35 and middle_up and ring_up and pinky_up:
        label, meaning, conf = \"OK\", \"OK / Perfect\", 0.9

    # HELLO / OPEN PALM — all 4 fingers up + thumb out (wave hello / number 5)
    elif index_up and middle_up and ring_up and pinky_up and thumb_extended:
        label, meaning, conf = \"HELLO\", \"Hello / Hi (wave) / 5\", 0.9

    # STOP / FLAT HAND / B — four fingers up, thumb across palm (not extended)
    elif index_up and middle_up and ring_up and pinky_up and not thumb_extended:
        label, meaning, conf = \"STOP\", \"Stop / Thank you / Flat hand\", 0.85

    # W — index + middle + ring up
    elif index_up and middle_up and ring_up and not pinky_up:
        label, meaning, conf = \"W / SIX\", \"Letter W / Number 6\", 0.85

    # POINT / 1 / D — only index up (no thumb)
    elif index_up and not middle_up and not ring_up and not pinky_up and not thumb_extended:
        label, meaning, conf = \"POINT\", \"Point / Number 1 / Letter D\", 0.85

    # L — thumb + index extended, others down
    elif thumb_extended and index_up and not middle_up and not ring_up and not pinky_up:
        label, meaning, conf = \"L\", \"Letter L / Loser\", 0.85

    # I — only pinky up
    elif pinky_up and not index_up and not middle_up and not ring_up and not thumb_extended:
        label, meaning, conf = \"I / LITTLE\", \"Letter I / Little finger\", 0.8

    # THUMBS UP / GOOD — only thumb extended vertically, fist underneath
    elif thumb_up_vertical and not index_up and not middle_up and not ring_up and not pinky_up:
        label, meaning, conf = \"GOOD\", \"Thumbs up / Good / Yes\", 0.9

    # FIST / S / NO — all curled
    elif not index_up and not middle_up and not ring_up and not pinky_up:
        if avg_curl < 0.95:
            if d_thumb_index < 0.55:
                label, meaning, conf = \"FIST\", \"Closed fist / Letter S / No\", 0.8
            else:
                # C / O — loose curl
                if d_thumb_index < 0.7:
                    label, meaning, conf = \"O\", \"Letter O / Zero\", 0.75
                else:
                    label, meaning, conf = \"C\", \"Letter C / Cup\", 0.7
        else:
            # Loose curl
            if d_thumb_index < 0.7:
                label, meaning, conf = \"O\", \"Letter O / Zero\", 0.7
            else:
                label, meaning, conf = \"C\", \"Letter C / Cup\", 0.7

    return RecognizeResponse(label=label, meaning=meaning, confidence=conf, debug=debug)


@api_router.get(\"/\")
async def root():
    return {\"message\": \"Sign Language Recognition API\", \"status\": \"ok\"}


@api_router.post(\"/recognize\", response_model=RecognizeResponse)
async def recognize(req: RecognizeRequest):
    return _classify(req.landmarks, req.handedness)


@api_router.get(\"/gestures\")
async def gestures():
    return {
        \"gestures\": [
            {\"label\": \"HELLO\",       \"meaning\": \"Hello / Hi (wave)\",     \"how\": \"Open palm, all 5 fingers spread\"},
            {\"label\": \"I LOVE YOU\",  \"meaning\": \"I love you\",            \"how\": \"Thumb + index + pinky out, middle & ring down\"},
            {\"label\": \"GOOD\",        \"meaning\": \"Thumbs up / Good / Yes\",\"how\": \"Only thumb pointing up\"},
            {\"label\": \"OK\",          \"meaning\": \"OK / Perfect\",          \"how\": \"Thumb-tip touches index-tip, others up\"},
            {\"label\": \"PEACE\",       \"meaning\": \"Peace / Victory\",       \"how\": \"Index + middle up in V\"},
            {\"label\": \"CALL ME\",     \"meaning\": \"Call me\",               \"how\": \"Thumb + pinky out, others curled\"},
            {\"label\": \"STOP\",        \"meaning\": \"Stop / Thank you\",      \"how\": \"Flat hand, 4 fingers up, thumb across palm\"},
            {\"label\": \"ROCK ON\",     \"meaning\": \"Rock on / Awesome\",     \"how\": \"Index + pinky up, thumb tucked\"},
            {\"label\": \"POINT\",       \"meaning\": \"Point / Number 1\",      \"how\": \"Only index finger up\"},
            {\"label\": \"THREE\",       \"meaning\": \"Number 3\",              \"how\": \"Thumb + index + middle up\"},
            {\"label\": \"FIST\",        \"meaning\": \"Closed fist / No\",      \"how\": \"All fingers curled tightly\"},
            {\"label\": \"L\",           \"meaning\": \"Letter L\",              \"how\": \"Thumb + index at 90°\"},
            {\"label\": \"C\",           \"meaning\": \"Letter C\",              \"how\": \"Curved hand like C\"},
            {\"label\": \"O\",           \"meaning\": \"Letter O / Zero\",       \"how\": \"Fingertips meet thumb in a circle\"},
            {\"label\": \"W / SIX\",     \"meaning\": \"W or 6\",                \"how\": \"Index + middle + ring up\"},
        ]
    }


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
