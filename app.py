import streamlit as st
import spacy
from spacy.tokens import Doc
from spacy import displacy

from spacy.language import Language
import re
from spacy.util import filter_spans

# Custom NER Component
class CustomNERComponent:
    def __init__(self, patterns):
        self.patterns = patterns

    def __call__(self, doc: Doc) -> Doc:
        # Create a list to hold the entities
        entities = []

        # Apply regex patterns
        for label, pattern_list in self.patterns.items():
            for pattern_info in pattern_list:
                pattern = pattern_info["pattern"]
                case_sensitive = pattern_info.get("case_sensitive", True)
                flags = 0 if case_sensitive else re.IGNORECASE

                for match in re.finditer(pattern, doc.text, flags):
                    start, end = match.span()
                    span = doc.char_span(start, end, label=label)
                    if span is not None:
                        entities.append(span)

        # Filter out overlapping spans
        entities = filter_spans(entities)

        # Assign non-overlapping entities to the doc
        doc.ents = entities
        return doc

# Load the spaCy model
nlp = spacy.load("model-best")

# Define your patterns (paste patterns from the code shared earlier)
patterns = {
    "SI_UNIT": [
    {
        "pattern": r'\b(?:meter|metre|m|kilometer|km|centimeter|cm|millimeter|mm|micrometer|µm|nanometer|nm|picometer|pm|decimeter|dm|femtometer|fm|attometer|am|light year|ly|astronomical unit|au|inch|foot|feet|ft|yard|yd|mile|nautical mile|nm|furlong|angstrom|Å|parsec|pc|newton|N|kilonewton|kN|joule|J|watt|W|pascal|Pa|hertz|Hz|coulomb|C|volt|V|ohm|Ω|siemens|S|farad|F|henry|H|lux|lx|becquerel|Bq|gray|Gy|sievert|Sv|liter|L|l|radian|rad|steradian|sr|dB|decibel)\b',
        "case_sensitive": True
    }
    ],
    "INTEGER": [{"pattern": r'\b\d+\b', "case_sensitive": True}],  # Integer
    "DECIMAL": [{"pattern": r'\b\d+\.\d+\b', "case_sensitive": True}],  # Decimal
    "NEGATIVE": [{"pattern": r'\b-\d+(\.\d+)?\b', "case_sensitive": True}],  # Negative
    "THOUSANDS_COMMA": [{"pattern": r'\b\d{1,3}(,\d{3})+\b', "case_sensitive": True}],  # Thousands with comma
    "THOUSANDS_SPACE": [{"pattern": r'\b\d{1,3}( \d{3})+\b', "case_sensitive": True}],  # Thousands with space
    "ORDINAL": [{"pattern": r'\b\d+(?:st|nd|rd|th)\b', "case_sensitive": True}],  # Ordinal
    "CARDINAL": [{"pattern": r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b', "case_sensitive": False}],  # Cardinal numbers
    "ROMAN": [{"pattern": r'\b[MCDLXVI]+\b', "case_sensitive": True}],  # Roman numerals
    "PERCENTAGE": [{"pattern": r'\b\d+%|\b\d+\spercent\b', "case_sensitive": True}],  # Percentage
    "RANGE_DASH": [{"pattern": r'\b\d+\s?[-–]\s?\d+\b', "case_sensitive": True}],  # Range with dash
    "RANGE_TO": [{"pattern": r'\b\d+\s?(?:to)\s?\d+\b', "case_sensitive": True}],  # Range with "to"
    "SCIENTIFIC": [{"pattern": r'\b\d+(\.\d+)?[eE][+-]?\d+\b', "case_sensitive": True}],  # Scientific notation
    "PHONE": [{"pattern": r'\b(?:\+?\d{1,3})?[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b', "case_sensitive": True}],  # Phone number
    "TIME": [{"pattern": r'\b\d{1,2}:\d{2}(:\d{2})?\b', "case_sensitive": True}],  # Time format
    "FRACTION": [{"pattern": r'\b\d+/\d+\b', "case_sensitive": True}],  # Fraction
    "VERSION": [{"pattern": r'\bv?\d+\.\d+(?:\.\d+)?\b', "case_sensitive": True}],  # Version number
    "DATE": [
        {"pattern": r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', "case_sensitive": True},  # MM/DD/YYYY
        {"pattern": r'\b\d{1,2}-\d{1,2}-\d{2,4}\b', "case_sensitive": True},  # DD-MM-YYYY
        {"pattern": r'\b\d{1,2}\s\w{3,9}\s\d{4}\b', "case_sensitive": True}  # DD Month YYYY
    ],
    "RANGE": [
        {"pattern": r'\b\d+\s?[-–]\s?\d+\b', "case_sensitive": True},  # Range with dash
        {"pattern": r'\b\d+\s?(?:to)\s?\d+\b', "case_sensitive": True}  # Range with "to"
    ],
    "TIME_UNIT": [
        {"pattern": r'\b\d{1,2}:\d{2}(:\d{2})?\b', "case_sensitive": True},  # Time format
        {"pattern": r'\b\d+\s?(?:hour|minute|second|s)\b', "case_sensitive": True}  # Time unit
    ],
    "COUNTABLE": [{"pattern": r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b', "case_sensitive": False}],  # Cardinal numbers
    "OPERATOR": [{"pattern": r'\b(?:[+\-*/=<>^%&|!~]+\b)', "case_sensitive": True}],  # Mathematical operators
    "EMAIL": [{"pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', "case_sensitive": True}],  # Email
    "INLINE_LIST": [{"pattern": r'\b(?:item\s+\d+)(?:\s?,\s?item\s+\d+)*\b', "case_sensitive": True}],  # Inline list
    "EQU": [{"pattern": r'\b(?:\w+)\s?=\s?\w+\b', "case_sensitive": True}],  # Equations
    "THOUSANDS_OPERATOR": [{"pattern": r'\b\d{1,3}(?:,\d{3})+\b', "case_sensitive": True}],  # Thousands with comma
    "ANGLE": [{"pattern": r'\b\d+°\b', "case_sensitive": True}],  # Angle in degrees
    "LEADING_ZERO": [{"pattern": r'\b0\d+\b', "case_sensitive": True}],  # Leading zero in number
    "RATIO": [{"pattern": r'\b\d+:\d+\b', "case_sensitive": True}] ,
    # Ratio
    "GREEK_LETTERS": [
        {"pattern": r'\b[Α-Ωα-ω]\b', "tag": "GREEK_LETTER", "case_sensitive": True}  # Greek letters
    ],
    
    "FIGURE_REFERENCE": [
        {
            "pattern": r'\b(?:[Ff]ig(?:ure)?|[Tt]able)[.,]?\s?\d+\b',
            "case_sensitive": False
        }
    ],
     "CITATION": [
        {
            "pattern": r'\b[A-Z][a-z]+ et al\. \(\d{4}\)\b',
            "case_sensitive": False
        }
    ],
    "YEAR": [
        {
            "pattern": r'\b\d{4}\b',
            "case_sensitive": True
        }
    ],


}

# Register the custom component using the @Language.factory decorator
if not Language.has_factory("custom_ner_component"):
    @Language.factory("custom_ner_component")
    def create_custom_ner_component(nlp, name):
        return CustomNERComponent(patterns)

# Add the custom component if not already in the pipeline
if "custom_ner_component" not in nlp.pipe_names:
    nlp.add_pipe("custom_ner_component", after="ner")

# Streamlit app interface
st.title("Custom NER MODEL Trained using Spacy and Rules Applied")
st.write("Enter some text for entity recognition:")

# Default text for the text box
default_text = """The study consisted of six numerical series – Figure 12 – with chord thicknesses varying from 3.0 to 5.5 mm with 0.5 mm increments. The brace diameter was also varied in each series, from values from 30 to 80 mm, with 10 mm increments"""

# Text input box with default value
user_input = st.text_area("Enter your text:", value=default_text)

# Button to trigger the analysis
if st.button("Analyze"):
    doc = nlp(user_input)

    # Display detected entities using displacy
    html = displacy.render(doc, style="ent", jupyter=False)
    st.write("Detected Entities:")
    st.components.v1.html(html, height=400)
