"""A worked Creative Generation example: the prompt in, the answer out, the scores.

Creative Generation is the worst case — 20 level-3 criteria across 6 groups, the
longest checklist, and the response most likely to outrun a token budget. A plain
photographic prompt makes most of those criteria inapplicable, which is exactly the
path worth showing: N/A is excluded from the mean, so a dimension can score well on
six criteria and say nothing about the other fourteen.

Used three ways: shown in the UI so people know what to expect before spending GPU
quota, offered as a fillable example, and as the fixture the parse-path tests run
against — which is why EXAMPLE_JSON is built from the checklist itself rather than
typed out, since a hand-written copy would quietly drift from the real criterion
names.
"""

from .scoring import CHECKLIST_L3_TO_L2

EXAMPLE_DIMENSION = "Creative Generation"

EXAMPLE_PROMPT = (
    "A red ceramic teapot pouring steaming tea into a white cup on a wooden "
    "table, morning light from a window on the left."
)

# What the judge actually concluded, keyed by level-3 criterion. Everything absent
# here is N/A: there is no text, no design brief and no narrative in a teapot photo.
EXAMPLE_VERDICTS = {
    "Imagination": 1,
    "Feature Matching": 1,
    "Logical Resolution": 2,
    "Cinematic Style": 1,
    "Camera / Lens Style": 1,
    "Composition": 2,
}


def _build_example_json():
    """Nest the verdicts under their real level-2 groups, N/A for the rest."""
    grouped = {}
    for level3, level2 in CHECKLIST_L3_TO_L2[EXAMPLE_DIMENSION].items():
        grouped.setdefault(level2, {})[level3] = {
            "score": EXAMPLE_VERDICTS.get(level3, "N/A")
        }
    return grouped


EXAMPLE_JSON = _build_example_json()

# Level-1 works out to 1.33: the mean of Imagination 1, Feature Matching 1, Logical
# Resolution 2 and Visual Storytelling 1.33, with Text Rendering and Design
# Applications dropping out entirely as all-N/A groups.
EXAMPLE_LEVEL1_SCORE = 4 / 3

EXAMPLE_OUTPUT = """\
The prompt asks for a red ceramic teapot pouring tea into a white cup, on wood, lit \
from a window on the left. This is a straightforward representational scene, so most \
of the creative checklist will not apply — I should be careful to mark those N/A \
rather than inventing a verdict for them.

Imagination: the scene is competently realised but conventional. It is a product-style \
still life of a kind that exists in thousands of stock photographs; there is no \
inventive concept, recombination or surprise. That is a pass rather than an excel — \
nothing is wrong, nothing is imaginative. Score 1.

Feature Matching: every element the prompt named is present and identifiable — the \
teapot reads as red ceramic, the cup is white, the table is wood, the steam is \
visible and the light does come from the left. It does what was asked without \
elaborating on it. Score 1.

Logical Resolution: this is where the image is genuinely strong. The pour is \
continuous from spout to cup surface, the stream has believable thickness and \
refraction, the steam rises from the liquid rather than floating free, and the \
shadows fall to the right consistently with a left-hand window. The contact between \
teapot base and table is clean. Score 2.

Text Accuracy, Text Layout, Font, Cross-lingual Generation: there is no text anywhere \
in the image and none was requested. All N/A.

Graphic Design, Product Design, Spatial Design, Fashion Styling, Game Design, Art \
Design: the prompt commissions a photograph, not a design artefact. None of these \
disciplines is being exercised, so judging them would be inventing a score. All N/A.

Cinematic Style: there is a mild warm-morning grade and soft directional light, which \
reads as a deliberate look, but it is not a distinctive cinematic treatment. Score 1.

Camera / Lens Style: shallow depth of field with the teapot spout sharp and the \
background falling off — consistent with a macro or short telephoto. Competent and \
unremarkable. Score 1.

Storyboard Creation, Shot Sizes, Angles, Comic Creation: no sequence, no specified \
framing or camera angle, no panel structure. All N/A.

Composition: the diagonal of the pour leads the eye from upper left to lower right, \
the cup sits near a third, and the negative space on the right is balanced against \
the mass of the teapot. This is well handled. Score 2.

So: six criteria scored, fourteen N/A.
</think>
{
  "Imagination": {"Imagination": {"score": 1}},
  "Feature Matching": {"Feature Matching": {"score": 1}},
  "Logical Resolution": {"Logical Resolution": {"score": 2}},
  "Text Rendering": {
    "Text Accuracy": {"score": "N/A"},
    "Text Layout": {"score": "N/A"},
    "Font": {"score": "N/A"},
    "Cross-lingual Generation": {"score": "N/A"}
  },
  "Design Applications": {
    "Graphic Design": {"score": "N/A"},
    "Product Design": {"score": "N/A"},
    "Spatial Design": {"score": "N/A"},
    "Fashion Styling": {"score": "N/A"},
    "Game Design": {"score": "N/A"},
    "Art Design": {"score": "N/A"}
  },
  "Visual Storytelling": {
    "Cinematic Style": {"score": 1},
    "Camera / Lens Style": {"score": 1},
    "Storyboard Creation": {"score": "N/A"},
    "Shot Sizes": {"score": "N/A"},
    "Composition": {"score": 2},
    "Angles": {"score": "N/A"},
    "Comic Creation": {"score": "N/A"}
  }
}"""
